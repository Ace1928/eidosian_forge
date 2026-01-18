import math
import warnings
from typing import Any, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import gather_params_ctx
from peft.utils.other import transpose
from .config import LoraConfig
class Linear(nn.Module, LoraLayer):

    def __init__(self, base_layer, adapter_name: str, r: int=0, lora_alpha: int=1, lora_dropout: float=0.0, fan_in_fan_out: bool=False, is_target_conv_1d_layer: bool=False, init_lora_weights: Union[bool, str]=True, use_rslora: bool=False, use_dora: bool=False, **kwargs) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, init_lora_weights=init_lora_weights, use_rslora=use_rslora, use_dora=use_dora)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool=False, adapter_names: Optional[List[str]]=None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return
        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        orig_weights += delta_weight
                    else:
                        weight_norm = self._get_weight_norm(orig_weights, delta_weight, scaling=1).detach()
                        self._cache_store(f'{active_adapter}-weight_norm', weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter] / weight_norm
                        orig_weights = dora_factor.view(-1, 1) * (orig_weights + delta_weight)
                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(f'NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken')
                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        base_layer.weight.data += delta_weight
                    else:
                        weight_norm = self._get_weight_norm(base_layer.weight, delta_weight, scaling=1).detach()
                        self._cache_store(f'{active_adapter}-weight_norm', weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter] / weight_norm
                        new_weight = dora_factor.view(-1, 1) * (base_layer.weight.data + delta_weight)
                        base_layer.weight.data = new_weight
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn('Already unmerged. Nothing to do.')
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight.data -= delta_weight
                else:
                    weight_norm = self._cache_pop(f'{active_adapter}-weight_norm')
                    dora_factor = self.lora_magnitude_vector[active_adapter] / weight_norm
                    weight_orig = weight.data / dora_factor.view(-1, 1) - delta_weight
                    weight.data = weight_orig

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype
        cast_to_fp32 = device.type == 'cpu' and dtype == torch.float16
        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight
        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()
        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]
        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)
        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)
                if not self.use_dora[active_adapter]:
                    result = result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    x = dropout(x)
                    result = result + self._apply_dora(x, lora_A, lora_B, scaling, active_adapter)
            result = result.to(torch_result_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return 'lora.' + rep
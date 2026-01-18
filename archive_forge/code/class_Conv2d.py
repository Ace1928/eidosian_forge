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
class Conv2d(nn.Module, LoraLayer):

    def __init__(self, base_layer: nn.Module, adapter_name: str, r: int=0, lora_alpha: int=1, lora_dropout: float=0.0, init_lora_weights: Union[bool, str]=True, use_rslora: bool=False, use_dora: bool=False, **kwargs) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer)
        if use_dora:
            raise ValueError(f'{self.__class__.__name__} does not support DoRA yet, please set it to False')
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, init_lora_weights=init_lora_weights, use_rslora=use_rslora, use_dora=use_dora)

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora):
        if r <= 0:
            raise ValueError(f'`r` should be a positive integer value but the value passed is {r}')
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()
        self.lora_dropout[adapter_name] = lora_dropout_layer
        base_layer = self.get_base_layer()
        kernel_size = base_layer.kernel_size
        stride = base_layer.stride
        padding = base_layer.padding
        self.lora_A[adapter_name] = nn.Conv2d(self.in_features, r, kernel_size, stride, padding, bias=False)
        self.lora_B[adapter_name] = nn.Conv2d(r, self.out_features, (1, 1), (1, 1), bias=False)
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights == 'loftq':
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        weight = getattr(base_layer, 'weight', None)
        if weight is not None:
            self.to(base_layer.weight.device, dtype=weight.dtype)
        self.set_adapter(self.active_adapters)

    def merge(self, safe_merge: bool=False, adapter_names: Optional[List[str]]=None) -> None:
        """
        Merge the active adapter weights inside the base weights

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
                    orig_weights += self.get_delta_weight(active_adapter)
                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(f'NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken')
                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
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
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_A[adapter].weight.dtype
        cast_to_fp32 = device.type == 'cpu' and dtype == torch.float16
        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight
        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()
        if self.get_base_layer().weight.size()[2:4] == (1, 1):
            output_tensor = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3) * self.scaling[adapter]
        else:
            output_tensor = F.conv2d(weight_A.permute(1, 0, 2, 3), weight_B).permute(1, 0, 2, 3) * self.scaling[adapter]
        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)
        return output_tensor

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
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
                result += lora_B(lora_A(dropout(x))) * scaling
            result = result.to(torch_result_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return 'lora.' + rep
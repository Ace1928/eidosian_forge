import importlib.metadata as importlib_metadata
from typing import Any, Optional
import packaging.version
import torch
from peft.import_utils import is_auto_awq_available
from peft.tuners.lora.layer import LoraLayer
from peft.tuners.tuners_utils import BaseTunerLayer
class AwqLoraLinear(torch.nn.Module, LoraLayer):

    def __init__(self, base_layer, adapter_name, r: int=0, lora_alpha: int=1, lora_dropout: float=0.0, init_lora_weights: bool=True, use_rslora: bool=False, **kwargs):
        super().__init__()
        LoraLayer.__init__(self, base_layer)
        self.quant_linear_module = base_layer
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora)

    def forward(self, x: torch.Tensor):
        result = self.quant_linear_module(x)
        if self.disable_adapters:
            return result
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                x = x.to(lora_A.weight.dtype)
            output = lora_B(lora_A(dropout(x)))
            if requires_conversion:
                output = output.to(expected_dtype)
            output = output * scaling
            result = result + output
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return 'lora.' + rep
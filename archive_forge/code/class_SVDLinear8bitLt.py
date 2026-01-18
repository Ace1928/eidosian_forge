from typing import Any
import torch
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from .layer import AdaLoraLayer
class SVDLinear8bitLt(torch.nn.Module, AdaLoraLayer):

    def __init__(self, base_layer: torch.nn.Module, adapter_name: str, r: int=0, lora_alpha: int=1, lora_dropout: float=0.0, init_lora_weights: bool=True, **kwargs) -> None:
        super().__init__()
        AdaLoraLayer.__init__(self, base_layer)
        self.get_base_layer().weight.requires_grad = False
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        if self.disable_adapters:
            return result
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                if x.dtype != torch.float32:
                    x = x.float()
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            lora_E = self.lora_E[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            ranknum = self.ranknum[active_adapter] + 1e-05
            output = dropout(x) @ (lora_A * lora_E).T @ lora_B.T
            if requires_conversion:
                output = output.to(expected_dtype)
            output = output * scaling / ranknum
            result = result + output
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return 'adalora.' + rep
import warnings
from typing import Any, List, Optional
import torch
from torch import nn
from peft.tuners.lora import LoraLayer
from peft.tuners.tuners_utils import check_adapters_to_merge
from peft.utils import transpose
class AdaLoraLayer(LoraLayer):
    adapter_layer_names = ('lora_A', 'lora_B', 'lora_E', 'lora_embedding_A', 'lora_embedding_B')

    def __init__(self, base_layer: nn.Module) -> None:
        super().__init__(base_layer)
        self.lora_E = nn.ParameterDict({})
        self.lora_A = nn.ParameterDict({})
        self.lora_B = nn.ParameterDict({})
        self.ranknum = nn.ParameterDict({})

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        if r <= 0:
            raise ValueError(f'`r` should be a positive integer value but the value passed is {r}')
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()
        self.lora_dropout[adapter_name] = lora_dropout_layer
        self.lora_A[adapter_name] = nn.Parameter(torch.randn(r, self.in_features))
        self.lora_E[adapter_name] = nn.Parameter(torch.randn(r, 1))
        self.lora_B[adapter_name] = nn.Parameter(torch.randn(self.out_features, r))
        self.ranknum[adapter_name] = nn.Parameter(torch.randn(1), requires_grad=False)
        self.ranknum[adapter_name].data.fill_(float(r))
        self.ranknum[adapter_name].requires_grad = False
        self.scaling[adapter_name] = lora_alpha if lora_alpha > 0 else float(r)
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        if hasattr(self.get_base_layer(), 'qweight'):
            self.to(self.get_base_layer().qweight.device)
        else:
            self.to(self.get_base_layer().weight.device)
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            nn.init.normal_(self.lora_E[adapter_name], mean=0.0, std=0.02)
            nn.init.normal_(self.lora_A[adapter_name], mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B[adapter_name], mean=0.0, std=0.02)
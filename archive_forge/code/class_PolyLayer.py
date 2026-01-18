import math
from typing import Any
import torch
import torch.nn as nn
from peft.tuners.tuners_utils import BaseTunerLayer
from .config import PolyConfig
from .router import get_router
class PolyLayer(BaseTunerLayer):
    adapter_layer_names = ('poly_lora_A', 'poly_lora_B', 'poly_router')
    other_param_names = ('r', 'n_tasks', 'n_skills', 'n_splits')

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.n_tasks = {}
        self.n_skills = {}
        self.n_splits = {}
        self.poly_type = {}
        self.poly_router = nn.ModuleDict()
        self.poly_lora_A = nn.ParameterDict()
        self.poly_lora_B = nn.ParameterDict()
        self.kwargs = kwargs
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = (base_layer.in_features, base_layer.out_features)
        else:
            raise ValueError(f'Unsupported layer type {type(base_layer)}')
        self.in_features = in_features
        self.out_features = out_features

    def update_layer(self, adapter_name, poly_config):
        if poly_config.r <= 0:
            raise ValueError(f'`r` should be a positive integer value but the value passed is {poly_config.r}')
        self.r[adapter_name] = poly_config.r
        self.n_tasks[adapter_name] = poly_config.n_tasks
        self.n_skills[adapter_name] = poly_config.n_skills
        self.n_splits[adapter_name] = poly_config.n_splits
        self.poly_type[adapter_name] = poly_config.poly_type
        self.poly_lora_A[adapter_name] = nn.Parameter(torch.empty(poly_config.n_splits, poly_config.n_skills, self.in_features // poly_config.n_splits, poly_config.r))
        self.poly_lora_B[adapter_name] = nn.Parameter(torch.empty(poly_config.n_splits, poly_config.n_skills, poly_config.r, self.out_features // poly_config.n_splits))
        self.poly_router[adapter_name] = get_router(poly_config)
        self.reset_poly_parameters(adapter_name, init_weights=poly_config.init_weights)
        weight = getattr(self.get_base_layer(), 'weight', None)
        if weight is not None:
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)

    def reset_poly_parameters(self, adapter_name, init_weights):
        if adapter_name in self.poly_lora_A.keys():
            n_splits, n_skills, d, r = self.poly_lora_A[adapter_name].shape
            for skill in range(n_skills):
                for split in range(n_splits):
                    param = torch.empty((r, d))
                    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                    self.poly_lora_A[adapter_name].data[split, skill, :, :] = param.T
            if init_weights:
                torch.nn.init.zeros_(self.poly_lora_B[adapter_name])
            else:
                n_splits, n_skills, r, d = self.poly_lora_B[adapter_name].shape
                for skill in range(n_skills):
                    for split in range(n_splits):
                        param = torch.empty((d, r))
                        torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                        self.poly_lora_B[adapter_name].data[split, skill, :, :] = param.T
            self.poly_router[adapter_name].reset()
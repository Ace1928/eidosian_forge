import math
from typing import Any, Set, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners.lycoris_utils import LycorisLayer
class LoHaLayer(nn.Module, LycorisLayer):
    adapter_layer_names = ('hada_w1_a', 'hada_w1_b', 'hada_w2_a', 'hada_w2_b', 'hada_t1', 'hada_t2')

    def __init__(self, base_layer: nn.Module):
        super().__init__()
        LycorisLayer.__init__(self, base_layer)
        self.hada_w1_a = nn.ParameterDict({})
        self.hada_w1_b = nn.ParameterDict({})
        self.hada_w2_a = nn.ParameterDict({})
        self.hada_w2_b = nn.ParameterDict({})
        self.hada_t1 = nn.ParameterDict({})
        self.hada_t2 = nn.ParameterDict({})

    @property
    def _available_adapters(self) -> Set[str]:
        return {*self.hada_w1_a, *self.hada_w1_b, *self.hada_w2_a, *self.hada_w2_b, *self.hada_t1, *self.hada_t2}

    def create_adapter_parameters(self, adapter_name: str, r: int, shape: Tuple[int, ...]):
        if len(shape) == 4:
            self.hada_t1[adapter_name] = nn.Parameter(torch.empty(r, r, shape[2], shape[3]))
            self.hada_w1_a[adapter_name] = nn.Parameter(torch.empty(r, shape[0]))
            self.hada_w1_b[adapter_name] = nn.Parameter(torch.empty(r, shape[1]))
            self.hada_t2[adapter_name] = nn.Parameter(torch.empty(r, r, shape[2], shape[3]))
            self.hada_w2_a[adapter_name] = nn.Parameter(torch.empty(r, shape[0]))
            self.hada_w2_b[adapter_name] = nn.Parameter(torch.empty(r, shape[1]))
        else:
            self.hada_w1_a[adapter_name] = nn.Parameter(torch.empty(shape[0], r))
            self.hada_w1_b[adapter_name] = nn.Parameter(torch.empty(r, shape[1]))
            self.hada_w2_a[adapter_name] = nn.Parameter(torch.empty(shape[0], r))
            self.hada_w2_b[adapter_name] = nn.Parameter(torch.empty(r, shape[1]))

    def reset_adapter_parameters(self, adapter_name: str):
        if adapter_name in self.hada_w1_a.keys():
            nn.init.kaiming_uniform_(self.hada_w1_a[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.hada_w1_b[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.hada_w2_a[adapter_name], a=math.sqrt(5))
            nn.init.zeros_(self.hada_w2_b[adapter_name])
        if adapter_name in self.hada_t1.keys():
            nn.init.kaiming_uniform_(self.hada_t1[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.hada_t2[adapter_name], a=math.sqrt(5))

    def reset_adapter_parameters_random(self, adapter_name: str):
        if adapter_name in self.hada_w1_a.keys():
            nn.init.kaiming_uniform_(self.hada_w1_a[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.hada_w1_b[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.hada_w2_a[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.hada_w2_b[adapter_name], a=math.sqrt(5))
        if adapter_name in self.hada_t1.keys():
            nn.init.kaiming_uniform_(self.hada_t1[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.hada_t2[adapter_name], a=math.sqrt(5))

    def update_layer(self, adapter_name: str, r: int, alpha: float, rank_dropout: float, module_dropout: float, init_weights: bool, use_effective_conv2d: bool=False, **kwargs) -> None:
        """Internal function to create loha adapter

        Args:
            adapter_name (`str`): Name for the adapter to add.
            r (`int`): Rank for the added adapter.
            alpha (`float`): Alpha for the added adapter.
            rank_dropout (`float`): The dropout probability for rank dimension during training.
            module_dropout (`float`): The dropout probability for disabling adapter during training.
            init_weights (`bool`): Whether to initialize weights.
            use_effective_conv2d (`bool`, *optional*, defaults to `False`):
                Use parameter effective decomposition for Conv2d with ksize > 1.
        """
        if r <= 0:
            raise ValueError(f'`r` should be a positive integer value but the value passed is {r}')
        self.r[adapter_name] = r
        self.alpha[adapter_name] = alpha
        self.scaling[adapter_name] = alpha / r
        self.rank_dropout[adapter_name] = rank_dropout
        self.module_dropout[adapter_name] = module_dropout
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            shape = tuple(base_layer.weight.shape)
        elif isinstance(base_layer, nn.Conv2d):
            use_effective_conv2d = use_effective_conv2d and base_layer.kernel_size != (1, 1)
            if use_effective_conv2d:
                shape = (base_layer.out_channels, base_layer.in_channels, *base_layer.kernel_size)
            else:
                shape = (base_layer.out_channels, base_layer.in_channels * base_layer.kernel_size[0] * base_layer.kernel_size[1])
        else:
            raise TypeError(f'LoHa is not implemented for base layers of type {type(base_layer).__name__}')
        self.create_adapter_parameters(adapter_name, r, shape)
        if init_weights:
            self.reset_adapter_parameters(adapter_name)
        else:
            self.reset_adapter_parameters_random(adapter_name)
        weight = getattr(self.get_base_layer(), 'weight', None)
        if weight is not None:
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        if adapter_name in self.hada_t1.keys():
            weight = make_weight_cp(self.hada_t1[adapter_name], self.hada_w1_a[adapter_name], self.hada_w1_b[adapter_name], self.hada_t2[adapter_name], self.hada_w2_a[adapter_name], self.hada_w2_b[adapter_name], scale=torch.tensor(self.scaling[adapter_name]))
        else:
            weight = make_weight(self.hada_w1_a[adapter_name], self.hada_w1_b[adapter_name], self.hada_w2_a[adapter_name], self.hada_w2_b[adapter_name], scale=torch.tensor(self.scaling[adapter_name]))
        base_layer = self.get_base_layer()
        weight = weight.reshape(base_layer.weight.shape)
        rank_dropout = self.rank_dropout[adapter_name]
        if self.training and rank_dropout:
            drop = (torch.rand(weight.size(0)) > rank_dropout).to(weight.dtype)
            drop = drop.view(-1, *[1] * len(weight.shape[1:])).to(weight.device)
            drop /= drop.mean()
            weight *= drop
        return weight

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self._available_adapters:
                    continue
                module_dropout = self.module_dropout[active_adapter]
                if not self.training or (self.training and torch.rand(1) > module_dropout):
                    result = result + self._get_delta_activations(active_adapter, x, *args, **kwargs)
        result = result.to(previous_dtype)
        return result
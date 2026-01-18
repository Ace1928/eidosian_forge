import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Union
import torch
from xformers.components import Activation
from xformers.components.feedforward import (
@register_feedforward('MixtureOfExperts', MoEConfig)
class MixtureOfExperts(Feedforward):
    """
        A MLP variant which uses the "Mixture of Experts" paradigm, as described in Gshard_.
        xFormers uses the FairScale_ implementation under the hood.

        .. warning: Please note that most of the benefits of MoE are present in a distributed training environmentt

        .. _Gshard: https://arxiv.org/pdf/2006.16668.pdf
        .. _FairScale: https://github.com/facebookresearch/fairscale/
        """

    def __init__(self, dim_model: int, dropout: float, activation: Activation, number_of_experts: int, gate: Union[GateConfig, torch.nn.Module], number_of_local_experts: Optional[int]=None, expert_constructor: Optional[Callable[[], torch.nn.Module]]=None, hidden_layer_multiplier: Optional[int]=None, group: Optional[Any]=None, *_, **__):
        super().__init__()
        assert dist.is_initialized(), 'Mixture of Experts require torch distributed to be initialized'
        if number_of_local_experts is not None:
            assert number_of_experts >= number_of_local_experts
        elif dist.get_world_size() == 1:
            logger.warning('Local experts no specified but world size of 1')
            logger.warning('Assuming that all experts are local')
            number_of_local_experts = number_of_experts
        else:
            number_of_local_experts = 1
        if not isinstance(gate, torch.nn.Module):
            gate_constructor = {GateConfig.RoundRobin: RoundRobinGate, GateConfig.Top2: Top2Gate}[gate]
            self.gate = gate_constructor(dim_model, number_of_experts)
        else:
            self.gate = gate
        if expert_constructor is None:
            multiplier = hidden_layer_multiplier if hidden_layer_multiplier is not None else 4

            def expert_constructor() -> torch.nn.Module:
                return MLP(dim_model, dropout, activation, multiplier)
            assert expert_constructor is not None
        local_experts = torch.nn.ModuleList([expert_constructor() for _ in range(number_of_local_experts)])
        self.moe = MOELayer(gate=self.gate, experts=local_experts, group=group)
        self.requires_cuda = True

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.moe(inputs.movedim(0, 1)).movedim(0, 1)
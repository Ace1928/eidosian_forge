import logging
import pathlib
from typing import (
from ray.rllib.core.learner.learner import (
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModule
from ray.rllib.core.rl_module.rl_module import (
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchDDPRLModule
from ray.rllib.core.rl_module.torch.torch_rl_module import (
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.annotations import (
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics import ALL_MODULES
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.torch_utils import (
from ray.rllib.utils.typing import Optimizer, Param, ParamDict, TensorType
@override(Learner)
def _check_registered_optimizer(self, optimizer: Optimizer, params: Sequence[Param]) -> None:
    super()._check_registered_optimizer(optimizer, params)
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError(f'The optimizer ({optimizer}) is not a torch.optim.Optimizer! Only use torch.optim.Optimizer subclasses for TorchLearner.')
    for param in params:
        if not isinstance(param, torch.Tensor):
            raise ValueError(f'One of the parameters ({param}) in the registered optimizer is not a torch.Tensor!')
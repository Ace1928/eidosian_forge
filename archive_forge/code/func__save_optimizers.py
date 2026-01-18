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
def _save_optimizers(self, path: Union[str, pathlib.Path]) -> None:
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)
    optim_state = self.get_optimizer_state()
    for name, state in optim_state.items():
        torch.save(state, path / f'{name}.pt')
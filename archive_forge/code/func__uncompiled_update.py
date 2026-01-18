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
def _uncompiled_update(self, batch: NestedDict, **kwargs):
    """Performs a single update given a batch of data."""
    fwd_out = self.module.forward_train(batch)
    loss_per_module = self.compute_loss(fwd_out=fwd_out, batch=batch)
    gradients = self.compute_gradients(loss_per_module)
    postprocessed_gradients = self.postprocess_gradients(gradients)
    self.apply_gradients(postprocessed_gradients)
    return (fwd_out, loss_per_module, self._metrics)
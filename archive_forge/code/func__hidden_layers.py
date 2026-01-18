import numpy as np
from typing import Dict, List
import gymnasium as gym
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import (
from ray.rllib.models.utils import get_activation_fn, get_filter_config
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
def _hidden_layers(self, obs: TensorType) -> TensorType:
    res = self._convs(obs.permute(0, 3, 1, 2))
    res = res.squeeze(3)
    res = res.squeeze(2)
    return res
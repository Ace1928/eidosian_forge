import abc
import logging
from typing import Tuple, Union
import numpy as np
from ray.rllib.core.models.base import Model
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.core.models.specs.checker import (
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType
from ray.util import log_once
@override(Model)
def _set_to_dummy_weights(self, value_sequence=(-0.02, -0.01, 0.01, 0.02)):
    trainable_weights = [p for p in self.parameters() if p.requires_grad]
    non_trainable_weights = [p for p in self.parameters() if not p.requires_grad]
    for i, w in enumerate(trainable_weights + non_trainable_weights):
        fill_val = value_sequence[i % len(value_sequence)]
        with torch.no_grad():
            w.fill_(fill_val)
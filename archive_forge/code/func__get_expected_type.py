import abc
from copy import deepcopy
import numpy as np
from typing import Any, Optional, Dict, List, Tuple, Union, Type
from ray.rllib.utils import try_import_jax, try_import_tf, try_import_torch
from ray.rllib.utils.annotations import OverrideToImplementCustomLogic
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.typing import TensorType
@OverrideToImplementCustomLogic
def _get_expected_type(self) -> Type:
    """Returns the expected type of the checked tensor."""
    if self._framework == 'torch':
        return torch.Tensor
    elif self._framework == 'tf2':
        return tf.Tensor
    elif self._framework == 'np':
        return np.ndarray
    elif self._framework == 'jax':
        jax, _ = try_import_jax()
        return jax.numpy.ndarray
    elif self._framework is None:
        return object
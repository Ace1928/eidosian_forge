import abc
from copy import deepcopy
import numpy as np
from typing import Any, Optional, Dict, List, Tuple, Union, Type
from ray.rllib.utils import try_import_jax, try_import_tf, try_import_torch
from ray.rllib.utils.annotations import OverrideToImplementCustomLogic
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.typing import TensorType
def _validate_shape_vals(self, d_names: List[str], shape_vals: Dict[str, int]) -> None:
    """Checks if the shape_vals is valid.

        Valid means that shape consist of unique dimension names and shape_vals only
        consists of keys that are in shape. Also shape_vals can only contain postive
        integers.
        """
    d_names_set = set(d_names)
    if len(d_names_set) != len(d_names):
        raise ValueError(_INVALID_INPUT_DUP_DIM.format(','.join(d_names)))
    for d_name in shape_vals:
        if d_name not in d_names_set:
            raise ValueError(_INVALID_INPUT_UNKNOWN_DIM.format(d_name, ','.join(d_names)))
        d_value = shape_vals.get(d_name, None)
        if d_value is not None:
            if not isinstance(d_value, int):
                raise ValueError(_INVALID_INPUT_INT_DIM.format(d_name, ','.join(d_names), type(d_value)))
            if d_value <= 0:
                raise ValueError(_INVALID_INPUT_POSITIVE.format(d_name, ','.join(d_names), d_value))
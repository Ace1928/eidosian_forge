import gc
import json
import os
import re
import warnings
from functools import partial
from pickle import UnpicklingError
from typing import Any, Dict, Optional, Set, Tuple, Union
import flax.linen as nn
import jax
import jax.numpy as jnp
import msgpack.exceptions
from flax.core.frozen_dict import FrozenDict, unfreeze
from flax.serialization import from_bytes, to_bytes
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.random import PRNGKey
from .configuration_utils import PretrainedConfig
from .dynamic_module_utils import custom_object_save
from .generation import FlaxGenerationMixin, GenerationConfig
from .modeling_flax_pytorch_utils import load_pytorch_checkpoint_in_flax_state_dict
from .utils import (
from .utils.hub import convert_file_size_to_int, get_checkpoint_shard_files
from .utils.import_utils import is_safetensors_available
def _cast_floating_to(self, params: Union[Dict, FrozenDict], dtype: jnp.dtype, mask: Any=None) -> Any:
    """
        Helper method to cast floating-point values of given parameter `PyTree` to given `dtype`.
        """

    def conditional_cast(param):
        if isinstance(param, jnp.ndarray) and jnp.issubdtype(param.dtype, jnp.floating):
            param = param.astype(dtype)
        return param
    if mask is None:
        return jax.tree_util.tree_map(conditional_cast, params)
    flat_params = flatten_dict(params)
    flat_mask, _ = jax.tree_util.tree_flatten(mask)
    for masked, key in zip(flat_mask, sorted(flat_params.keys())):
        if masked:
            flat_params[key] = conditional_cast(flat_params[key])
    return unflatten_dict(flat_params)
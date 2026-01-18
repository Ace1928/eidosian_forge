import copy
import json
import os
import warnings
from collections import UserDict
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union
import numpy as np
from .dynamic_module_utils import custom_object_save
from .utils import (
def _get_is_as_tensor_fns(self, tensor_type: Optional[Union[str, TensorType]]=None):
    if tensor_type is None:
        return (None, None)
    if not isinstance(tensor_type, TensorType):
        tensor_type = TensorType(tensor_type)
    if tensor_type == TensorType.TENSORFLOW:
        if not is_tf_available():
            raise ImportError('Unable to convert output to TensorFlow tensors format, TensorFlow is not installed.')
        import tensorflow as tf
        as_tensor = tf.constant
        is_tensor = tf.is_tensor
    elif tensor_type == TensorType.PYTORCH:
        if not is_torch_available():
            raise ImportError('Unable to convert output to PyTorch tensors format, PyTorch is not installed.')
        import torch

        def as_tensor(value):
            if isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], np.ndarray):
                value = np.array(value)
            return torch.tensor(value)
        is_tensor = torch.is_tensor
    elif tensor_type == TensorType.JAX:
        if not is_flax_available():
            raise ImportError('Unable to convert output to JAX tensors format, JAX is not installed.')
        import jax.numpy as jnp
        as_tensor = jnp.array
        is_tensor = is_jax_tensor
    else:

        def as_tensor(value, dtype=None):
            if isinstance(value, (list, tuple)) and isinstance(value[0], (list, tuple, np.ndarray)):
                value_lens = [len(val) for val in value]
                if len(set(value_lens)) > 1 and dtype is None:
                    value = as_tensor([np.asarray(val) for val in value], dtype=object)
            return np.asarray(value, dtype=dtype)
        is_tensor = is_numpy_array
    return (is_tensor, as_tensor)
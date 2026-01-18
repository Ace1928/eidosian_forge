import logging
import os
from typing import List
import numpy as np
import ray
from ray.util.collective import types
def _check_tensor_lists_input(tensor_lists):
    """Check if the input is a list of lists of supported tensor types."""
    if not isinstance(tensor_lists, list):
        raise RuntimeError("The input must be a list of lists of tensors. Got '{}'.".format(type(tensor_lists)))
    if not tensor_lists:
        raise RuntimeError(f'Did not receive tensors. Got: {tensor_lists}')
    for t in tensor_lists:
        _check_tensor_list_input(t)
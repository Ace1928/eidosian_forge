import logging
import os
from typing import List
import numpy as np
import ray
from ray.util.collective import types
def _check_root_tensor_valid(length, root_tensor):
    """Check the root_tensor device is 0 <= root_tensor < length"""
    if root_tensor < 0:
        raise ValueError("root_tensor '{}' is negative.".format(root_tensor))
    if root_tensor >= length:
        raise ValueError("root_tensor '{}' is greater than the number of GPUs: '{}'".format(root_tensor, length))
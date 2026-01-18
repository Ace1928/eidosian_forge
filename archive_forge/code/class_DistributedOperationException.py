import pickle
import warnings
from functools import update_wrapper, wraps
from typing import Any, Mapping
import torch
from ..state import PartialState
from .constants import TORCH_DISTRIBUTED_OPERATION_TYPES
from .dataclasses import DistributedType, TensorInformation
from .imports import (
class DistributedOperationException(Exception):
    """
    An exception class for distributed operations. Raised if the operation cannot be performed due to the shape of the
    tensors.
    """
    pass
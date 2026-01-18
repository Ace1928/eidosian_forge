import collections
import copy
import functools
import itertools
import operator
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List
import torch
from transformers.file_utils import add_end_docstrings
from transformers.utils.fx import _gen_constructor_wrapper
@staticmethod
def _get_bias(linear: torch.nn.Linear) -> torch.Tensor:
    if linear.bias is not None:
        return linear.bias
    return torch.zeros(linear.out_features, dtype=linear.weight.dtype).to(linear.weight.device)
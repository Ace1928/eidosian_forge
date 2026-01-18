import contextlib
import ctypes
import glob
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, TypeVar, cast
import torch
from .common import BaseOperator, get_operator, get_xformers_operator, register_operator
@register_operator
class SparsifyBothWays(BaseOperator):
    OPERATOR = get_xformers_operator('sparse24_sparsify_both_ways')
    OPERATOR_CATEGORY = 'sp24'
    NAME = 'sparse24_sparsify_both_ways'
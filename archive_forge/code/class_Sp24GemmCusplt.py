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
class Sp24GemmCusplt(BaseOperator):
    OPERATOR = get_operator('aten', '_cslt_sparse_mm')
    OPERATOR_CATEGORY = 'sp24'
    NAME = f'_cslt_sparse_mm@{_cusplt_version_str}'
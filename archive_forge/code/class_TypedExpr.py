import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, overload, Tuple, Union
import sympy
from typing_extensions import TypeAlias
import torch
from torch._prims_common import is_boolean_dtype, is_integer_dtype
from torch.utils._sympy.functions import FloorDiv, ModularIndexing, Where
@dataclass
class TypedExpr:
    """A SymPy expression with associated type"""
    expr: sympy.Expr
    dtype: torch.dtype
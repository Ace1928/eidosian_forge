import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, overload, Tuple, Union
import sympy
from typing_extensions import TypeAlias
import torch
from torch._prims_common import is_boolean_dtype, is_integer_dtype
from torch.utils._sympy.functions import FloorDiv, ModularIndexing, Where
@dataclass
class IndexPropVar:
    value: Any
    is_symbolic: bool = False

    @staticmethod
    def new_symbolic(expr: TypedExpr) -> 'IndexPropVar':
        return IndexPropVar(expr, is_symbolic=True)

    def __post_init__(self):
        assert not self.is_symbolic or isinstance(self.value, TypedExpr), 'Symbolic IndexPropVar must contain a TypedExpr'
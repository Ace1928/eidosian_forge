import collections
import dataclasses
import itertools
import logging
import re
import typing
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import sympy
import torch
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from .codegen.common import index_prevent_reordering
from .utils import get_dtype_size, sympy_str, sympy_subs, sympy_symbol, VarRanges
from .virtualized import V
class WeakDep(typing.NamedTuple):
    name: str

    @property
    def index(self):
        raise NotImplementedError('WeakDep does not have an index')

    def get_numel(self) -> sympy.Expr:
        return sympy.Integer(1)

    def rename(self, renames: Dict[str, str]) -> 'WeakDep':
        if self.name in renames:
            return WeakDep(renames[self.name])
        return self

    def numbytes_hint(self):
        return 1

    def has_unbacked_symbols(self):
        return False

    def is_contiguous(self) -> bool:
        return False
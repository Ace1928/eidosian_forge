import ast
import builtins
import operator
from collections import ChainMap, OrderedDict, deque
from contextlib import suppress
from types import FrameType
from typing import Any, Tuple, Iterable, List, Mapping, Dict, Union, Set
from pure_eval.my_getattr_static import getattr_static
from pure_eval.utils import (
def _handle_subscript(self, node):
    value = self[node.value]
    of_standard_types(value, check_dict_values=False, deep=is_any(type(value), dict, OrderedDict))
    index = node.slice
    if isinstance(index, ast.Slice):
        index = slice(*[None if p is None else self[p] for p in [index.lower, index.upper, index.step]])
    elif isinstance(index, ast.ExtSlice):
        raise CannotEval
    else:
        if isinstance(index, ast.Index):
            index = index.value
        index = self[index]
    of_standard_types(index, check_dict_values=False, deep=True)
    try:
        return value[index]
    except Exception:
        raise CannotEval
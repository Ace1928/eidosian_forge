import collections
from collections import defaultdict
from .node import Node, Argument, Target, map_arg, _type_repr, _get_qualified_name
import torch.utils._pytree as pytree
from . import _pytree as fx_pytree
from ._compatibility import compatibility
import contextlib
from typing import TYPE_CHECKING, Callable, Any, List, Dict, NamedTuple, Optional, Tuple, Set, FrozenSet, Type
from dataclasses import dataclass
from contextlib import contextmanager
import copy
import enum
import torch
import keyword
import re
import builtins
import math
import warnings
import inspect
def _get_repr(arg: Any) -> str:
    if isinstance(arg, tuple) and hasattr(arg, '_fields'):
        qualified_name = _get_qualified_name(type(arg))
        global_name = add_global(qualified_name, type(arg))
        return f'{global_name}{repr(tuple(arg))}'
    elif isinstance(arg, torch._ops.OpOverload):
        qualified_name = _get_qualified_name(arg)
        global_name = add_global(qualified_name, arg)
        return f'{global_name}'
    elif isinstance(arg, enum.Enum):
        cls = arg.__class__
        clsname = add_global(cls.__name__, cls)
        return f'{clsname}.{arg.name}'
    return repr(arg)
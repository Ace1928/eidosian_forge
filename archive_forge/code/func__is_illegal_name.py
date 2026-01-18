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
def _is_illegal_name(self, name: str, obj: Any) -> bool:
    if name in keyword.kwlist:
        return True
    if name in builtins.__dict__:
        return obj is not builtins.__dict__[name]
    if name in _custom_builtins:
        return obj is not _custom_builtins[name].obj
    return False
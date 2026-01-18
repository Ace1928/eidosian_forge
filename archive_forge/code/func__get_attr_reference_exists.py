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
def _get_attr_reference_exists(mod: torch.nn.Module, qualified_name: str) -> bool:
    module_path, _, name = qualified_name.rpartition('.')
    try:
        submod: torch.nn.Module = mod.get_submodule(module_path)
    except AttributeError:
        warnings.warn(f'Failed to fetch module {module_path}!')
        return False
    if not hasattr(submod, name):
        return False
    res = getattr(submod, name)
    if not isinstance(res, torch.nn.Module) and (not isinstance(res, torch.nn.Parameter)) and (name not in submod._buffers):
        return False
    return True
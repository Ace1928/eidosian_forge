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
class _PyTreeInfo(NamedTuple):
    """
    Contains extra info stored when we're using Pytrees
    """
    orig_args: List[str]
    in_spec: pytree.TreeSpec
    out_spec: Optional[pytree.TreeSpec]
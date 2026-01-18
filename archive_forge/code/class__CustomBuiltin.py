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
class _CustomBuiltin(NamedTuple):
    """Additional objs that we add to every graph's globals.

    The repr() for some standard library objects is not valid Python code without
    an import. For common objects of this sort, we bundle them in the globals of
    every FX graph.
    """
    import_str: str
    obj: Any
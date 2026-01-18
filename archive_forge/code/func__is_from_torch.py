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
def _is_from_torch(obj: Any) -> bool:
    module_name = getattr(obj, '__module__', None)
    if module_name is not None:
        base_module = module_name.partition('.')[0]
        return base_module == 'torch' and (not module_name.startswith('torch._dynamo.')) and (not module_name.startswith('torch._inductor.'))
    name = getattr(obj, '__name__', None)
    if name is not None and name != 'torch':
        for guess in [torch, torch.nn.functional]:
            if getattr(guess, name, None) is obj:
                return True
    return False
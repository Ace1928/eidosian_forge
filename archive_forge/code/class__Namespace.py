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
class _Namespace:
    """A context for associating names uniquely with objects.

    The following invariants are enforced:
    - Each object gets a single name.
    - Each name is unique within a given namespace.
    - Names generated do not shadow builtins, unless the object is indeed that builtin.
    """

    def __init__(self):
        self._obj_to_name: Dict[Any, str] = {}
        self._unassociated_names = set()
        self._used_names: Set[str] = set()
        self._base_count: Dict[str, int] = defaultdict(int)
        self._illegal_char_regex = re.compile('[^0-9a-zA-Z_]+')
        self._name_suffix_regex = re.compile('(.*)_(\\d+)$')

    def create_name(self, candidate: str, obj: Optional[Any]) -> str:
        """Create a unique name.

        Arguments:
            candidate: used as the basis for the unique name, relevant to the user.
            obj: If not None, an object that will be associated with the unique name.
        """
        if obj is not None and obj in self._obj_to_name:
            return self._obj_to_name[obj]
        candidate = self._illegal_char_regex.sub('_', candidate)
        if not candidate:
            candidate = '_unnamed'
        if candidate[0].isdigit():
            candidate = f'_{candidate}'
        match = self._name_suffix_regex.match(candidate)
        if match is None:
            base = candidate
            num = None
        else:
            base, num_str = match.group(1, 2)
            num = int(num_str)
        candidate = base if num is None else f'{base}_{num}'
        if not num:
            num = self._base_count[base]
        while candidate in self._used_names or self._is_illegal_name(candidate, obj):
            num += 1
            candidate = f'{base}_{num}'
        self._used_names.add(candidate)
        self._base_count[base] = num
        if obj is None:
            self._unassociated_names.add(candidate)
        else:
            self._obj_to_name[obj] = candidate
        return candidate

    def associate_name_with_obj(self, name: str, obj: Any):
        """Associate a unique name with an object.

        Neither `name` nor `obj` should be associated already.
        """
        assert obj not in self._obj_to_name
        assert name in self._unassociated_names
        self._obj_to_name[obj] = name
        self._unassociated_names.remove(name)

    def _is_illegal_name(self, name: str, obj: Any) -> bool:
        if name in keyword.kwlist:
            return True
        if name in builtins.__dict__:
            return obj is not builtins.__dict__[name]
        if name in _custom_builtins:
            return obj is not _custom_builtins[name].obj
        return False

    def _rename_object(self, obj: Any, name: str):
        assert obj in self._obj_to_name
        self._obj_to_name[obj] = name
        self._used_names.add(name)
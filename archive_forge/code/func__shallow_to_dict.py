from __future__ import annotations
from collections import deque
import collections.abc as collections_abc
import itertools
from itertools import zip_longest
import operator
import typing
from typing import Any
from typing import Callable
from typing import Deque
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from . import operators
from .cache_key import HasCacheKey
from .visitors import _TraverseInternalsType
from .visitors import anon_map
from .visitors import ExternallyTraversible
from .visitors import HasTraversalDispatch
from .visitors import HasTraverseInternals
from .. import util
from ..util import langhelpers
from ..util.typing import Self
def _shallow_to_dict(self) -> Dict[str, Any]:
    cls = self.__class__
    shallow_to_dict: Callable[[HasShallowCopy], Dict[str, Any]]
    try:
        shallow_to_dict = cls.__dict__['_generated_shallow_to_dict_traversal']
    except KeyError:
        shallow_to_dict = self._generate_shallow_to_dict(cls._traverse_internals, '_generated_shallow_to_dict_traversal')
        cls._generated_shallow_to_dict_traversal = shallow_to_dict
    return shallow_to_dict(self)
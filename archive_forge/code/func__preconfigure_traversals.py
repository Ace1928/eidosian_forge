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
def _preconfigure_traversals(target_hierarchy: Type[Any]) -> None:
    for cls in util.walk_subclasses(target_hierarchy):
        if hasattr(cls, '_generate_cache_attrs') and hasattr(cls, '_traverse_internals'):
            cls._generate_cache_attrs()
            _copy_internals.generate_dispatch(cls, cls._traverse_internals, '_generated_copy_internals_traversal')
            _get_children.generate_dispatch(cls, cls._traverse_internals, '_generated_get_children_traversal')
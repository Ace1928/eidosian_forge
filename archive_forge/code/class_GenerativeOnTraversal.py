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
class GenerativeOnTraversal(HasShallowCopy):
    """Supplies Generative behavior but making use of traversals to shallow
    copy.

    .. seealso::

        :class:`sqlalchemy.sql.base.Generative`


    """
    __slots__ = ()

    def _generate(self) -> Self:
        cls = self.__class__
        s = cls.__new__(cls)
        self._shallow_copy_to(s)
        return s
from __future__ import annotations
from collections import deque
from enum import Enum
import itertools
import operator
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import exc
from .. import util
from ..util import langhelpers
from ..util._has_cy import HAS_CYEXTENSION
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
class HasTraverseInternals:
    """base for classes that have a "traverse internals" element,
    which defines all kinds of ways of traversing the elements of an object.

    Compared to :class:`.Visitable`, which relies upon an external visitor to
    define how the object is travered (i.e. the :class:`.SQLCompiler`), the
    :class:`.HasTraverseInternals` interface allows classes to define their own
    traversal, that is, what attributes are accessed and in what order.

    """
    __slots__ = ()
    _traverse_internals: _TraverseInternalsType
    _is_immutable: bool = False

    @util.preload_module('sqlalchemy.sql.traversals')
    def get_children(self, *, omit_attrs: Tuple[str, ...]=(), **kw: Any) -> Iterable[HasTraverseInternals]:
        """Return immediate child :class:`.visitors.HasTraverseInternals`
        elements of this :class:`.visitors.HasTraverseInternals`.

        This is used for visit traversal.

        \\**kw may contain flags that change the collection that is
        returned, for example to return a subset of items in order to
        cut down on larger traversals, or to return child items from a
        different context (such as schema-level collections instead of
        clause-level).

        """
        traversals = util.preloaded.sql_traversals
        try:
            traverse_internals = self._traverse_internals
        except AttributeError:
            return []
        dispatch = traversals._get_children.run_generated_dispatch
        return itertools.chain.from_iterable((meth(obj, **kw) for attrname, obj, meth in dispatch(self, traverse_internals, '_generated_get_children_traversal') if attrname not in omit_attrs and obj is not None))
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
class ExternalTraversal(util.MemoizedSlots):
    """Base class for visitor objects which can traverse externally using
    the :func:`.visitors.traverse` function.

    Direct usage of the :func:`.visitors.traverse` function is usually
    preferred.

    """
    __slots__ = ('_visitor_dict', '_next')
    __traverse_options__: Dict[str, Any] = {}
    _next: Optional[ExternalTraversal]

    def traverse_single(self, obj: Visitable, **kw: Any) -> Any:
        for v in self.visitor_iterator:
            meth = getattr(v, 'visit_%s' % obj.__visit_name__, None)
            if meth:
                return meth(obj, **kw)

    def iterate(self, obj: Optional[ExternallyTraversible]) -> Iterator[ExternallyTraversible]:
        """Traverse the given expression structure, returning an iterator
        of all elements.

        """
        return iterate(obj, self.__traverse_options__)

    @overload
    def traverse(self, obj: Literal[None]) -> None:
        ...

    @overload
    def traverse(self, obj: ExternallyTraversible) -> ExternallyTraversible:
        ...

    def traverse(self, obj: Optional[ExternallyTraversible]) -> Optional[ExternallyTraversible]:
        """Traverse and visit the given expression structure."""
        return traverse(obj, self.__traverse_options__, self._visitor_dict)

    def _memoized_attr__visitor_dict(self) -> Dict[str, _TraverseCallableType[Any]]:
        visitors = {}
        for name in dir(self):
            if name.startswith('visit_'):
                visitors[name[6:]] = getattr(self, name)
        return visitors

    @property
    def visitor_iterator(self) -> Iterator[ExternalTraversal]:
        """Iterate through this visitor and each 'chained' visitor."""
        v: Optional[ExternalTraversal] = self
        while v:
            yield v
            v = getattr(v, '_next', None)

    def chain(self: _ExtT, visitor: ExternalTraversal) -> _ExtT:
        """'Chain' an additional ExternalTraversal onto this ExternalTraversal

        The chained visitor will receive all visit events after this one.

        """
        tail = list(self.visitor_iterator)[-1]
        tail._next = visitor
        return self
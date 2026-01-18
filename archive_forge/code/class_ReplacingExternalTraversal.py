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
class ReplacingExternalTraversal(CloningExternalTraversal):
    """Base class for visitor objects which can traverse using
    the :func:`.visitors.replacement_traverse` function.

    Direct usage of the :func:`.visitors.replacement_traverse` function is
    usually preferred.

    """
    __slots__ = ()

    def replace(self, elem: ExternallyTraversible) -> Optional[ExternallyTraversible]:
        """Receive pre-copied elements during a cloning traversal.

        If the method returns a new element, the element is used
        instead of creating a simple copy of the element.  Traversal
        will halt on the newly returned element if it is re-encountered.
        """
        return None

    @overload
    def traverse(self, obj: Literal[None]) -> None:
        ...

    @overload
    def traverse(self, obj: ExternallyTraversible) -> ExternallyTraversible:
        ...

    def traverse(self, obj: Optional[ExternallyTraversible]) -> Optional[ExternallyTraversible]:
        """Traverse and visit the given expression structure."""

        def replace(element: ExternallyTraversible, **kw: Any) -> Optional[ExternallyTraversible]:
            for v in self.visitor_iterator:
                e = cast(ReplacingExternalTraversal, v).replace(element)
                if e is not None:
                    return e
            return None
        return replacement_traverse(obj, self.__traverse_options__, replace)
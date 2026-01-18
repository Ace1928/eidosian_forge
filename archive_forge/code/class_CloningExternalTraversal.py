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
class CloningExternalTraversal(ExternalTraversal):
    """Base class for visitor objects which can traverse using
    the :func:`.visitors.cloned_traverse` function.

    Direct usage of the :func:`.visitors.cloned_traverse` function is usually
    preferred.


    """
    __slots__ = ()

    def copy_and_process(self, list_: List[ExternallyTraversible]) -> List[ExternallyTraversible]:
        """Apply cloned traversal to the given list of elements, and return
        the new list.

        """
        return [self.traverse(x) for x in list_]

    @overload
    def traverse(self, obj: Literal[None]) -> None:
        ...

    @overload
    def traverse(self, obj: ExternallyTraversible) -> ExternallyTraversible:
        ...

    def traverse(self, obj: Optional[ExternallyTraversible]) -> Optional[ExternallyTraversible]:
        """Traverse and visit the given expression structure."""
        return cloned_traverse(obj, self.__traverse_options__, self._visitor_dict)
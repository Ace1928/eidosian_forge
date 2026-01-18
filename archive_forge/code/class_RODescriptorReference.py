from __future__ import annotations
import builtins
import collections.abc as collections_abc
import re
import sys
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import ForwardRef
from typing import Generic
from typing import Iterable
from typing import Mapping
from typing import NewType
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import compat
class RODescriptorReference(Generic[_DESC_co]):
    """a descriptor that refers to a descriptor.

    same as :class:`.DescriptorReference` but is read-only, so that subclasses
    can define a subtype as the generically contained element

    """
    if TYPE_CHECKING:

        def __get__(self, instance: object, owner: Any) -> _DESC_co:
            ...

        def __set__(self, instance: Any, value: Any) -> NoReturn:
            ...

        def __delete__(self, instance: Any) -> NoReturn:
            ...
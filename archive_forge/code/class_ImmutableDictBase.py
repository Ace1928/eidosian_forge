from __future__ import annotations
from itertools import filterfalse
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from ..util.typing import Self
class ImmutableDictBase(ReadOnlyContainer, Dict[_KT, _VT]):
    if TYPE_CHECKING:

        def __new__(cls, *args: Any) -> Self:
            ...

        def __init__(cls, *args: Any):
            ...

    def _readonly(self, *arg: Any, **kw: Any) -> NoReturn:
        self._immutable()

    def clear(self) -> NoReturn:
        self._readonly()

    def pop(self, key: Any, default: Optional[Any]=None) -> NoReturn:
        self._readonly()

    def popitem(self) -> NoReturn:
        self._readonly()

    def setdefault(self, key: Any, default: Optional[Any]=None) -> NoReturn:
        self._readonly()

    def update(self, *arg: Any, **kw: Any) -> NoReturn:
        self._readonly()
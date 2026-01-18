from __future__ import annotations
from enum import Enum
import functools
import itertools
import operator
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .row import Row
from .row import RowMapping
from .. import exc
from .. import util
from ..sql.base import _generative
from ..sql.base import HasMemoized
from ..sql.base import InPlaceGenerative
from ..util import HasMemoized_ro_memoized_attribute
from ..util import NONE_SET
from ..util._has_cy import HAS_CYEXTENSION
from ..util.typing import Literal
from ..util.typing import Self
class RMKeyView(typing.KeysView[Any]):
    __slots__ = ('_parent', '_keys')
    _parent: ResultMetaData
    _keys: Sequence[str]

    def __init__(self, parent: ResultMetaData):
        self._parent = parent
        self._keys = [k for k in parent._keys if k is not None]

    def __len__(self) -> int:
        return len(self._keys)

    def __repr__(self) -> str:
        return '{0.__class__.__name__}({0._keys!r})'.format(self)

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __contains__(self, item: Any) -> bool:
        if isinstance(item, int):
            return False
        return self._parent._has_key(item)

    def __eq__(self, other: Any) -> bool:
        return list(other) == list(self)

    def __ne__(self, other: Any) -> bool:
        return list(other) != list(self)
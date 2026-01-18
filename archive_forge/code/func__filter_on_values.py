from __future__ import annotations
from abc import ABC
import collections.abc as collections_abc
import operator
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from ..sql import util as sql_util
from ..util import deprecated
from ..util._has_cy import HAS_CYEXTENSION
def _filter_on_values(self, processor: Optional[_ProcessorsType]) -> Row[Any]:
    return Row(self._parent, processor, self._key_to_index, self._data)
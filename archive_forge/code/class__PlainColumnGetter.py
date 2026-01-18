from __future__ import annotations
import operator
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import base
from .collections import collection
from .collections import collection_adapter
from .. import exc as sa_exc
from .. import util
from ..sql import coercions
from ..sql import expression
from ..sql import roles
from ..util.typing import Literal
class _PlainColumnGetter(Generic[_KT]):
    """Plain column getter, stores collection of Column objects
    directly.

    Serializes to a :class:`._SerializableColumnGetterV2`
    which has more expensive __call__() performance
    and some rare caveats.

    """
    __slots__ = ('cols', 'composite')

    def __init__(self, cols: Sequence[ColumnElement[_KT]]) -> None:
        self.cols = cols
        self.composite = len(cols) > 1

    def __reduce__(self) -> Tuple[Type[_SerializableColumnGetterV2[_KT]], Tuple[Sequence[Tuple[Optional[str], Optional[str]]]]]:
        return _SerializableColumnGetterV2._reduce_from_cols(self.cols)

    def _cols(self, mapper: Mapper[_KT]) -> Sequence[ColumnElement[_KT]]:
        return self.cols

    def __call__(self, value: _KT) -> Union[_KT, Tuple[_KT, ...]]:
        state = base.instance_state(value)
        m = base._state_mapper(state)
        key: List[_KT] = [m._get_state_attr_by_column(state, state.dict, col) for col in self._cols(m)]
        if self.composite:
            return tuple(key)
        else:
            obj = key[0]
            if obj is None:
                return _UNMAPPED_AMBIGUOUS_NONE
            else:
                return obj
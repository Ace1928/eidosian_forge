from __future__ import annotations
import typing
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union
from ..util.typing import Literal
class cache_anon_map(Dict[Union[int, 'Literal[CacheConst.NO_CACHE]'], Union[Literal[True], str]]):
    """A map that creates new keys for missing key access.

    Produces an incrementing sequence given a series of unique keys.

    This is similar to the compiler prefix_anon_map class although simpler.

    Inlines the approach taken by :class:`sqlalchemy.util.PopulateDict` which
    is otherwise usually used for this type of operation.

    """
    _index = 0

    def get_anon(self, object_: Any) -> Tuple[str, bool]:
        idself = id(object_)
        if idself in self:
            s_val = self[idself]
            assert s_val is not True
            return (s_val, True)
        else:
            self[idself] = id_ = str(self._index)
            self._index += 1
            return (id_, False)

    def __missing__(self, key: int) -> str:
        self[key] = val = str(self._index)
        self._index += 1
        return val
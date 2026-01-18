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
def _union_w_kw(self, __d: Optional[Mapping[_KT, _VT]]=None, **kw: _VT) -> immutabledict[_KT, _VT]:
    if not __d and (not kw):
        return self
    new = ImmutableDictBase.__new__(self.__class__)
    dict.__init__(new, self)
    if __d:
        dict.update(new, __d)
    dict.update(new, kw)
    return new
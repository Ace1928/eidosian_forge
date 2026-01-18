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
class immutabledict(ImmutableDictBase[_KT, _VT]):

    def __new__(cls, *args):
        new = ImmutableDictBase.__new__(cls)
        dict.__init__(new, *args)
        return new

    def __init__(self, *args: Union[Mapping[_KT, _VT], Iterable[Tuple[_KT, _VT]]]):
        pass

    def __reduce__(self):
        return (immutabledict, (dict(self),))

    def union(self, __d: Optional[Mapping[_KT, _VT]]=None) -> immutabledict[_KT, _VT]:
        if not __d:
            return self
        new = ImmutableDictBase.__new__(self.__class__)
        dict.__init__(new, self)
        dict.update(new, __d)
        return new

    def _union_w_kw(self, __d: Optional[Mapping[_KT, _VT]]=None, **kw: _VT) -> immutabledict[_KT, _VT]:
        if not __d and (not kw):
            return self
        new = ImmutableDictBase.__new__(self.__class__)
        dict.__init__(new, self)
        if __d:
            dict.update(new, __d)
        dict.update(new, kw)
        return new

    def merge_with(self, *dicts: Optional[Mapping[_KT, _VT]]) -> immutabledict[_KT, _VT]:
        new = None
        for d in dicts:
            if d:
                if new is None:
                    new = ImmutableDictBase.__new__(self.__class__)
                    dict.__init__(new, self)
                dict.update(new, d)
        if new is None:
            return self
        return new

    def __repr__(self) -> str:
        return 'immutabledict(%s)' % dict.__repr__(self)

    def __ior__(self, __value: Any) -> NoReturn:
        self._readonly()

    def __or__(self, __value: Mapping[_KT, _VT]) -> immutabledict[_KT, _VT]:
        return immutabledict(super().__or__(__value))

    def __ror__(self, __value: Mapping[_KT, _VT]) -> immutabledict[_KT, _VT]:
        return immutabledict(super().__ror__(__value))
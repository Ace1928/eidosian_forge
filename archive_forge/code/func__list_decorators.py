from __future__ import annotations
import operator
import threading
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import List
from typing import NoReturn
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from .base import NO_KEY
from .. import exc as sa_exc
from .. import util
from ..sql.base import NO_ARG
from ..util.compat import inspect_getfullargspec
from ..util.typing import Protocol
def _list_decorators() -> Dict[str, Callable[[_FN], _FN]]:
    """Tailored instrumentation wrappers for any list-like class."""

    def _tidy(fn):
        fn._sa_instrumented = True
        fn.__doc__ = getattr(list, fn.__name__).__doc__

    def append(fn):

        def append(self, item, _sa_initiator=None):
            item = __set(self, item, _sa_initiator, NO_KEY)
            fn(self, item)
        _tidy(append)
        return append

    def remove(fn):

        def remove(self, value, _sa_initiator=None):
            __del(self, value, _sa_initiator, NO_KEY)
            fn(self, value)
        _tidy(remove)
        return remove

    def insert(fn):

        def insert(self, index, value):
            value = __set(self, value, None, index)
            fn(self, index, value)
        _tidy(insert)
        return insert

    def __setitem__(fn):

        def __setitem__(self, index, value):
            if not isinstance(index, slice):
                existing = self[index]
                if existing is not None:
                    __del(self, existing, None, index)
                value = __set(self, value, None, index)
                fn(self, index, value)
            else:
                step = index.step or 1
                start = index.start or 0
                if start < 0:
                    start += len(self)
                if index.stop is not None:
                    stop = index.stop
                else:
                    stop = len(self)
                if stop < 0:
                    stop += len(self)
                if step == 1:
                    if value is self:
                        return
                    for i in range(start, stop, step):
                        if len(self) > start:
                            del self[start]
                    for i, item in enumerate(value):
                        self.insert(i + start, item)
                else:
                    rng = list(range(start, stop, step))
                    if len(value) != len(rng):
                        raise ValueError('attempt to assign sequence of size %s to extended slice of size %s' % (len(value), len(rng)))
                    for i, item in zip(rng, value):
                        self.__setitem__(i, item)
        _tidy(__setitem__)
        return __setitem__

    def __delitem__(fn):

        def __delitem__(self, index):
            if not isinstance(index, slice):
                item = self[index]
                __del(self, item, None, index)
                fn(self, index)
            else:
                for item in self[index]:
                    __del(self, item, None, index)
                fn(self, index)
        _tidy(__delitem__)
        return __delitem__

    def extend(fn):

        def extend(self, iterable):
            for value in list(iterable):
                self.append(value)
        _tidy(extend)
        return extend

    def __iadd__(fn):

        def __iadd__(self, iterable):
            for value in list(iterable):
                self.append(value)
            return self
        _tidy(__iadd__)
        return __iadd__

    def pop(fn):

        def pop(self, index=-1):
            __before_pop(self)
            item = fn(self, index)
            __del(self, item, None, index)
            return item
        _tidy(pop)
        return pop

    def clear(fn):

        def clear(self, index=-1):
            for item in self:
                __del(self, item, None, index)
            fn(self)
        _tidy(clear)
        return clear
    l = locals().copy()
    l.pop('_tidy')
    return l
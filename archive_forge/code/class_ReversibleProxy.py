from __future__ import annotations
import abc
import functools
from typing import Any
from typing import AsyncGenerator
from typing import AsyncIterator
from typing import Awaitable
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import Generator
from typing import Generic
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Tuple
from typing import TypeVar
import weakref
from . import exc as async_exc
from ... import util
from ...util.typing import Literal
from ...util.typing import Self
class ReversibleProxy(Generic[_PT]):
    _proxy_objects: ClassVar[Dict[weakref.ref[Any], weakref.ref[ReversibleProxy[Any]]]] = {}
    __slots__ = ('__weakref__',)

    @overload
    def _assign_proxied(self, target: _PT) -> _PT:
        ...

    @overload
    def _assign_proxied(self, target: None) -> None:
        ...

    def _assign_proxied(self, target: Optional[_PT]) -> Optional[_PT]:
        if target is not None:
            target_ref: weakref.ref[_PT] = weakref.ref(target, ReversibleProxy._target_gced)
            proxy_ref = weakref.ref(self, functools.partial(ReversibleProxy._target_gced, target_ref))
            ReversibleProxy._proxy_objects[target_ref] = proxy_ref
        return target

    @classmethod
    def _target_gced(cls, ref: weakref.ref[_PT], proxy_ref: Optional[weakref.ref[Self]]=None) -> None:
        cls._proxy_objects.pop(ref, None)

    @classmethod
    def _regenerate_proxy_for_target(cls, target: _PT) -> Self:
        raise NotImplementedError()

    @overload
    @classmethod
    def _retrieve_proxy_for_target(cls, target: _PT, regenerate: Literal[True]=...) -> Self:
        ...

    @overload
    @classmethod
    def _retrieve_proxy_for_target(cls, target: _PT, regenerate: bool=True) -> Optional[Self]:
        ...

    @classmethod
    def _retrieve_proxy_for_target(cls, target: _PT, regenerate: bool=True) -> Optional[Self]:
        try:
            proxy_ref = cls._proxy_objects[weakref.ref(target)]
        except KeyError:
            pass
        else:
            proxy = proxy_ref()
            if proxy is not None:
                return proxy
        if regenerate:
            return cls._regenerate_proxy_for_target(target)
        else:
            return None
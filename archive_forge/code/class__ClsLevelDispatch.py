from __future__ import annotations
import collections
from itertools import chain
import threading
from types import TracebackType
import typing
from typing import Any
from typing import cast
from typing import Collection
from typing import Deque
from typing import FrozenSet
from typing import Generic
from typing import Iterator
from typing import MutableMapping
from typing import MutableSequence
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
import weakref
from . import legacy
from . import registry
from .registry import _ET
from .registry import _EventKey
from .registry import _ListenerFnType
from .. import exc
from .. import util
from ..util.concurrency import AsyncAdaptedLock
from ..util.typing import Protocol
class _ClsLevelDispatch(RefCollection[_ET]):
    """Class-level events on :class:`._Dispatch` classes."""
    __slots__ = ('clsname', 'name', 'arg_names', 'has_kw', 'legacy_signatures', '_clslevel', '__weakref__')
    clsname: str
    name: str
    arg_names: Sequence[str]
    has_kw: bool
    legacy_signatures: MutableSequence[legacy._LegacySignatureType]
    _clslevel: MutableMapping[Type[_ET], _ListenerFnSequenceType[_ListenerFnType]]

    def __init__(self, parent_dispatch_cls: Type[_HasEventsDispatch[_ET]], fn: _ListenerFnType):
        self.name = fn.__name__
        self.clsname = parent_dispatch_cls.__name__
        argspec = util.inspect_getfullargspec(fn)
        self.arg_names = argspec.args[1:]
        self.has_kw = bool(argspec.varkw)
        self.legacy_signatures = list(reversed(sorted(getattr(fn, '_legacy_signatures', []), key=lambda s: s[0])))
        fn.__doc__ = legacy._augment_fn_docs(self, parent_dispatch_cls, fn)
        self._clslevel = weakref.WeakKeyDictionary()

    def _adjust_fn_spec(self, fn: _ListenerFnType, named: bool) -> _ListenerFnType:
        if named:
            fn = self._wrap_fn_for_kw(fn)
        if self.legacy_signatures:
            try:
                argspec = util.get_callable_argspec(fn, no_self=True)
            except TypeError:
                pass
            else:
                fn = legacy._wrap_fn_for_legacy(self, fn, argspec)
        return fn

    def _wrap_fn_for_kw(self, fn: _ListenerFnType) -> _ListenerFnType:

        def wrap_kw(*args: Any, **kw: Any) -> Any:
            argdict = dict(zip(self.arg_names, args))
            argdict.update(kw)
            return fn(**argdict)
        return wrap_kw

    def _do_insert_or_append(self, event_key: _EventKey[_ET], is_append: bool) -> None:
        target = event_key.dispatch_target
        assert isinstance(target, type), 'Class-level Event targets must be classes.'
        if not getattr(target, '_sa_propagate_class_events', True):
            raise exc.InvalidRequestError(f"Can't assign an event directly to the {target} class")
        cls: Type[_ET]
        for cls in util.walk_subclasses(target):
            if cls is not target and cls not in self._clslevel:
                self.update_subclass(cls)
            else:
                if cls not in self._clslevel:
                    self.update_subclass(cls)
                if is_append:
                    self._clslevel[cls].append(event_key._listen_fn)
                else:
                    self._clslevel[cls].appendleft(event_key._listen_fn)
        registry._stored_in_collection(event_key, self)

    def insert(self, event_key: _EventKey[_ET], propagate: bool) -> None:
        self._do_insert_or_append(event_key, is_append=False)

    def append(self, event_key: _EventKey[_ET], propagate: bool) -> None:
        self._do_insert_or_append(event_key, is_append=True)

    def update_subclass(self, target: Type[_ET]) -> None:
        if target not in self._clslevel:
            if getattr(target, '_sa_propagate_class_events', True):
                self._clslevel[target] = collections.deque()
            else:
                self._clslevel[target] = _empty_collection()
        clslevel = self._clslevel[target]
        cls: Type[_ET]
        for cls in target.__mro__[1:]:
            if cls in self._clslevel:
                clslevel.extend([fn for fn in self._clslevel[cls] if fn not in clslevel])

    def remove(self, event_key: _EventKey[_ET]) -> None:
        target = event_key.dispatch_target
        cls: Type[_ET]
        for cls in util.walk_subclasses(target):
            if cls in self._clslevel:
                self._clslevel[cls].remove(event_key._listen_fn)
        registry._removed_from_collection(event_key, self)

    def clear(self) -> None:
        """Clear all class level listeners"""
        to_clear: Set[_ListenerFnType] = set()
        for dispatcher in self._clslevel.values():
            to_clear.update(dispatcher)
            dispatcher.clear()
        registry._clear(self, to_clear)

    def for_modify(self, obj: _Dispatch[_ET]) -> _ClsLevelDispatch[_ET]:
        """Return an event collection which can be modified.

        For _ClsLevelDispatch at the class level of
        a dispatcher, this returns self.

        """
        return self
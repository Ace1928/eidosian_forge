from __future__ import annotations
import typing
from typing import Any
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import Union
import weakref
from .attr import _ClsLevelDispatch
from .attr import _EmptyListener
from .attr import _InstanceLevelDispatch
from .attr import _JoinedListener
from .registry import _ET
from .registry import _EventKey
from .. import util
from ..util.typing import Literal
class _Dispatch(_DispatchCommon[_ET]):
    """Mirror the event listening definitions of an Events class with
    listener collections.

    Classes which define a "dispatch" member will return a
    non-instantiated :class:`._Dispatch` subclass when the member
    is accessed at the class level.  When the "dispatch" member is
    accessed at the instance level of its owner, an instance
    of the :class:`._Dispatch` class is returned.

    A :class:`._Dispatch` class is generated for each :class:`.Events`
    class defined, by the :meth:`._HasEventsDispatch._create_dispatcher_class`
    method.  The original :class:`.Events` classes remain untouched.
    This decouples the construction of :class:`.Events` subclasses from
    the implementation used by the event internals, and allows
    inspecting tools like Sphinx to work in an unsurprising
    way against the public API.

    """
    __slots__ = ('_parent', '_instance_cls', '__dict__', '_empty_listeners')
    _active_history: bool
    _empty_listener_reg: MutableMapping[Type[_ET], Dict[str, _EmptyListener[_ET]]] = weakref.WeakKeyDictionary()
    _empty_listeners: Dict[str, _EmptyListener[_ET]]
    _event_names: List[str]
    _instance_cls: Optional[Type[_ET]]
    _joined_dispatch_cls: Type[_JoinedDispatcher[_ET]]
    _events: Type[_HasEventsDispatch[_ET]]
    'reference back to the Events class.\n\n    Bidirectional against _HasEventsDispatch.dispatch\n\n    '

    def __init__(self, parent: Optional[_Dispatch[_ET]], instance_cls: Optional[Type[_ET]]=None):
        self._parent = parent
        self._instance_cls = instance_cls
        if instance_cls:
            assert parent is not None
            try:
                self._empty_listeners = self._empty_listener_reg[instance_cls]
            except KeyError:
                self._empty_listeners = self._empty_listener_reg[instance_cls] = {ls.name: _EmptyListener(ls, instance_cls) for ls in parent._event_descriptors}
        else:
            self._empty_listeners = {}

    def __getattr__(self, name: str) -> _InstanceLevelDispatch[_ET]:
        try:
            ls = self._empty_listeners[name]
        except KeyError:
            raise AttributeError(name)
        else:
            setattr(self, ls.name, ls)
            return ls

    @property
    def _event_descriptors(self) -> Iterator[_ClsLevelDispatch[_ET]]:
        for k in self._event_names:
            yield getattr(self, k)

    def _listen(self, event_key: _EventKey[_ET], **kw: Any) -> None:
        return self._events._listen(event_key, **kw)

    def _for_class(self, instance_cls: Type[_ET]) -> _Dispatch[_ET]:
        return self.__class__(self, instance_cls)

    def _for_instance(self, instance: _ET) -> _Dispatch[_ET]:
        instance_cls = instance.__class__
        return self._for_class(instance_cls)

    def _join(self, other: _DispatchCommon[_ET]) -> _JoinedDispatcher[_ET]:
        """Create a 'join' of this :class:`._Dispatch` and another.

        This new dispatcher will dispatch events to both
        :class:`._Dispatch` objects.

        """
        if '_joined_dispatch_cls' not in self.__class__.__dict__:
            cls = type('Joined%s' % self.__class__.__name__, (_JoinedDispatcher,), {'__slots__': self._event_names})
            self.__class__._joined_dispatch_cls = cls
        return self._joined_dispatch_cls(self, other)

    def __reduce__(self) -> Union[str, Tuple[Any, ...]]:
        return (_UnpickleDispatch(), (self._instance_cls,))

    def _update(self, other: _Dispatch[_ET], only_propagate: bool=True) -> None:
        """Populate from the listeners in another :class:`_Dispatch`
        object."""
        for ls in other._event_descriptors:
            if isinstance(ls, _EmptyListener):
                continue
            getattr(self, ls.name).for_modify(self)._update(ls, only_propagate=only_propagate)

    def _clear(self) -> None:
        for ls in self._event_descriptors:
            ls.for_modify(self).clear()
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
class _HasEventsDispatch(Generic[_ET]):
    _dispatch_target: Optional[Type[_ET]]
    'class which will receive the .dispatch collection'
    dispatch: _Dispatch[_ET]
    'reference back to the _Dispatch class.\n\n    Bidirectional against _Dispatch._events\n\n    '
    if typing.TYPE_CHECKING:

        def __getattr__(self, name: str) -> _InstanceLevelDispatch[_ET]:
            ...

    def __init_subclass__(cls) -> None:
        """Intercept new Event subclasses and create associated _Dispatch
        classes."""
        cls._create_dispatcher_class(cls.__name__, cls.__bases__, cls.__dict__)

    @classmethod
    def _accept_with(cls, target: Union[_ET, Type[_ET]], identifier: str) -> Optional[Union[_ET, Type[_ET]]]:
        raise NotImplementedError()

    @classmethod
    def _listen(cls, event_key: _EventKey[_ET], *, propagate: bool=False, insert: bool=False, named: bool=False, asyncio: bool=False) -> None:
        raise NotImplementedError()

    @staticmethod
    def _set_dispatch(klass: Type[_HasEventsDispatch[_ET]], dispatch_cls: Type[_Dispatch[_ET]]) -> _Dispatch[_ET]:
        klass.dispatch = dispatch_cls(None)
        dispatch_cls._events = klass
        return klass.dispatch

    @classmethod
    def _create_dispatcher_class(cls, classname: str, bases: Tuple[type, ...], dict_: Mapping[str, Any]) -> None:
        """Create a :class:`._Dispatch` class corresponding to an
        :class:`.Events` class."""
        if hasattr(cls, 'dispatch'):
            dispatch_base = cls.dispatch.__class__
        else:
            dispatch_base = _Dispatch
        event_names = [k for k in dict_ if _is_event_name(k)]
        dispatch_cls = cast('Type[_Dispatch[_ET]]', type('%sDispatch' % classname, (dispatch_base,), {'__slots__': event_names}))
        dispatch_cls._event_names = event_names
        dispatch_inst = cls._set_dispatch(cls, dispatch_cls)
        for k in dispatch_cls._event_names:
            setattr(dispatch_inst, k, _ClsLevelDispatch(cls, dict_[k]))
            _registrars[k].append(cls)
        for super_ in dispatch_cls.__bases__:
            if issubclass(super_, _Dispatch) and super_ is not _Dispatch:
                for ls in super_._events.dispatch._event_descriptors:
                    setattr(dispatch_inst, ls.name, ls)
                    dispatch_cls._event_names.append(ls.name)
        if getattr(cls, '_dispatch_target', None):
            dispatch_target_cls = cls._dispatch_target
            assert dispatch_target_cls is not None
            if hasattr(dispatch_target_cls, '__slots__') and '_slots_dispatch' in dispatch_target_cls.__slots__:
                dispatch_target_cls.dispatch = slots_dispatcher(cls)
            else:
                dispatch_target_cls.dispatch = dispatcher(cls)
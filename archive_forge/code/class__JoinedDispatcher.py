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
class _JoinedDispatcher(_DispatchCommon[_ET]):
    """Represent a connection between two _Dispatch objects."""
    __slots__ = ('local', 'parent', '_instance_cls')
    local: _DispatchCommon[_ET]
    parent: _DispatchCommon[_ET]
    _instance_cls: Optional[Type[_ET]]

    def __init__(self, local: _DispatchCommon[_ET], parent: _DispatchCommon[_ET]):
        self.local = local
        self.parent = parent
        self._instance_cls = self.local._instance_cls

    def __getattr__(self, name: str) -> _JoinedListener[_ET]:
        ls = getattr(self.local, name)
        jl = _JoinedListener(self.parent, ls.name, ls)
        setattr(self, ls.name, jl)
        return jl

    def _listen(self, event_key: _EventKey[_ET], **kw: Any) -> None:
        return self.parent._listen(event_key, **kw)

    @property
    def _events(self) -> Type[_HasEventsDispatch[_ET]]:
        return self.parent._events
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
class _DispatchCommon(Generic[_ET]):
    __slots__ = ()
    _instance_cls: Optional[Type[_ET]]

    def _join(self, other: _DispatchCommon[_ET]) -> _JoinedDispatcher[_ET]:
        raise NotImplementedError()

    def __getattr__(self, name: str) -> _InstanceLevelDispatch[_ET]:
        raise NotImplementedError()

    @property
    def _events(self) -> Type[_HasEventsDispatch[_ET]]:
        raise NotImplementedError()
from __future__ import annotations
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
import weakref
from . import base
from . import exc as orm_exc
from . import interfaces
from ._typing import _O
from ._typing import is_collection_impl
from .base import ATTR_WAS_SET
from .base import INIT_OK
from .base import LoaderCallableStatus
from .base import NEVER_SET
from .base import NO_VALUE
from .base import PASSIVE_NO_INITIALIZE
from .base import PASSIVE_NO_RESULT
from .base import PASSIVE_OFF
from .base import SQL_OK
from .path_registry import PathRegistry
from .. import exc as sa_exc
from .. import inspection
from .. import util
from ..util.typing import Literal
from ..util.typing import Protocol
@classmethod
def _instance_level_callable_processor(cls, manager: ClassManager[_O], fn: _LoaderCallable, key: Any) -> _InstallLoaderCallableProto[_O]:
    impl = manager[key].impl
    if is_collection_impl(impl):
        fixed_impl = impl

        def _set_callable(state: InstanceState[_O], dict_: _InstanceDict, row: Row[Any]) -> None:
            if 'callables' not in state.__dict__:
                state.callables = {}
            old = dict_.pop(key, None)
            if old is not None:
                fixed_impl._invalidate_collection(old)
            state.callables[key] = fn
    else:

        def _set_callable(state: InstanceState[_O], dict_: _InstanceDict, row: Row[Any]) -> None:
            if 'callables' not in state.__dict__:
                state.callables = {}
            state.callables[key] = fn
    return _set_callable
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
def _expire(self, dict_: _InstanceDict, modified_set: Set[InstanceState[Any]]) -> None:
    self.expired = True
    if self.modified:
        modified_set.discard(self)
        self.committed_state.clear()
        self.modified = False
    self._strong_obj = None
    if '_pending_mutations' in self.__dict__:
        del self.__dict__['_pending_mutations']
    if 'parents' in self.__dict__:
        del self.__dict__['parents']
    self.expired_attributes.update([impl.key for impl in self.manager._loader_impls])
    if self.callables:
        for k in self.expired_attributes.intersection(self.callables):
            del self.callables[k]
    for k in self.manager._collection_impl_keys.intersection(dict_):
        collection = dict_.pop(k)
        collection._sa_adapter.invalidated = True
    if self._last_known_values:
        self._last_known_values.update({k: dict_[k] for k in self._last_known_values if k in dict_})
    for key in self.manager._all_key_set.intersection(dict_):
        del dict_[key]
    self.manager.dispatch.expire(self, None)
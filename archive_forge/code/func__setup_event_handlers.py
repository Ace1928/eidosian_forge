from __future__ import annotations
from dataclasses import is_dataclass
import inspect
import itertools
import operator
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import attributes
from . import util as orm_util
from .base import _DeclarativeMapped
from .base import LoaderCallableStatus
from .base import Mapped
from .base import PassiveFlag
from .base import SQLORMOperations
from .interfaces import _AttributeOptions
from .interfaces import _IntrospectsAnnotations
from .interfaces import _MapsColumns
from .interfaces import MapperProperty
from .interfaces import PropComparator
from .util import _none_set
from .util import de_stringify_annotation
from .. import event
from .. import exc as sa_exc
from .. import schema
from .. import sql
from .. import util
from ..sql import expression
from ..sql import operators
from ..sql.elements import BindParameter
from ..util.typing import is_fwd_ref
from ..util.typing import is_pep593
from ..util.typing import typing_get_args
def _setup_event_handlers(self) -> None:
    """Establish events that populate/expire the composite attribute."""

    def load_handler(state: InstanceState[Any], context: ORMCompileState) -> None:
        _load_refresh_handler(state, context, None, is_refresh=False)

    def refresh_handler(state: InstanceState[Any], context: ORMCompileState, to_load: Optional[Sequence[str]]) -> None:
        if not to_load or {self.key}.union(self._attribute_keys).intersection(to_load):
            _load_refresh_handler(state, context, to_load, is_refresh=True)

    def _load_refresh_handler(state: InstanceState[Any], context: ORMCompileState, to_load: Optional[Sequence[str]], is_refresh: bool) -> None:
        dict_ = state.dict
        if (not is_refresh or context is self._COMPOSITE_FGET) and self.key in dict_:
            return
        for k in self._attribute_keys:
            if k not in dict_:
                return
        dict_[self.key] = self.composite_class(*[state.dict[key] for key in self._attribute_keys])

    def expire_handler(state: InstanceState[Any], keys: Optional[Sequence[str]]) -> None:
        if keys is None or set(self._attribute_keys).intersection(keys):
            state.dict.pop(self.key, None)

    def insert_update_handler(mapper: Mapper[Any], connection: Connection, state: InstanceState[Any]) -> None:
        """After an insert or update, some columns may be expired due
            to server side defaults, or re-populated due to client side
            defaults.  Pop out the composite value here so that it
            recreates.

            """
        state.dict.pop(self.key, None)
    event.listen(self.parent, 'after_insert', insert_update_handler, raw=True)
    event.listen(self.parent, 'after_update', insert_update_handler, raw=True)
    event.listen(self.parent, 'load', load_handler, raw=True, propagate=True)
    event.listen(self.parent, 'refresh', refresh_handler, raw=True, propagate=True)
    event.listen(self.parent, 'expire', expire_handler, raw=True, propagate=True)
    proxy_attr = self.parent.class_manager[self.key]
    proxy_attr.impl.dispatch = proxy_attr.dispatch
    proxy_attr.impl.dispatch._active_history = self.active_history
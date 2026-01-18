from __future__ import annotations
import dataclasses
import operator
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import Iterable
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import collections
from . import exc as orm_exc
from . import interfaces
from ._typing import insp_is_aliased_class
from .base import _DeclarativeMapped
from .base import ATTR_EMPTY
from .base import ATTR_WAS_SET
from .base import CALLABLES_OK
from .base import DEFERRED_HISTORY_LOAD
from .base import INCLUDE_PENDING_MUTATIONS  # noqa
from .base import INIT_OK
from .base import instance_dict as instance_dict
from .base import instance_state as instance_state
from .base import instance_str
from .base import LOAD_AGAINST_COMMITTED
from .base import LoaderCallableStatus
from .base import manager_of_class as manager_of_class
from .base import Mapped as Mapped  # noqa
from .base import NEVER_SET  # noqa
from .base import NO_AUTOFLUSH
from .base import NO_CHANGE  # noqa
from .base import NO_KEY
from .base import NO_RAISE
from .base import NO_VALUE
from .base import NON_PERSISTENT_OK  # noqa
from .base import opt_manager_of_class as opt_manager_of_class
from .base import PASSIVE_CLASS_MISMATCH  # noqa
from .base import PASSIVE_NO_FETCH
from .base import PASSIVE_NO_FETCH_RELATED  # noqa
from .base import PASSIVE_NO_INITIALIZE
from .base import PASSIVE_NO_RESULT
from .base import PASSIVE_OFF
from .base import PASSIVE_ONLY_PERSISTENT
from .base import PASSIVE_RETURN_NO_VALUE
from .base import PassiveFlag
from .base import RELATED_OBJECT_OK  # noqa
from .base import SQL_OK  # noqa
from .base import SQLORMExpression
from .base import state_str
from .. import event
from .. import exc
from .. import inspection
from .. import util
from ..event import dispatcher
from ..event import EventTarget
from ..sql import base as sql_base
from ..sql import cache_key
from ..sql import coercions
from ..sql import roles
from ..sql import visitors
from ..sql.cache_key import HasCacheKey
from ..sql.visitors import _TraverseInternalsType
from ..sql.visitors import InternalTraversal
from ..util.typing import Literal
from ..util.typing import Self
from ..util.typing import TypeGuard
class InstrumentedAttribute(QueryableAttribute[_T_co]):
    """Class bound instrumented attribute which adds basic
    :term:`descriptor` methods.

    See :class:`.QueryableAttribute` for a description of most features.


    """
    __slots__ = ()
    inherit_cache = True
    ':meta private:'

    @util.rw_hybridproperty
    def __doc__(self) -> Optional[str]:
        return self._doc

    @__doc__.setter
    def __doc__(self, value: Optional[str]) -> None:
        self._doc = value

    @__doc__.classlevel
    def __doc__(cls) -> Optional[str]:
        return super().__doc__

    def __set__(self, instance: object, value: Any) -> None:
        self.impl.set(instance_state(instance), instance_dict(instance), value, None)

    def __delete__(self, instance: object) -> None:
        self.impl.delete(instance_state(instance), instance_dict(instance))

    @overload
    def __get__(self, instance: None, owner: Any) -> InstrumentedAttribute[_T_co]:
        ...

    @overload
    def __get__(self, instance: object, owner: Any) -> _T_co:
        ...

    def __get__(self, instance: Optional[object], owner: Any) -> Union[InstrumentedAttribute[_T_co], _T_co]:
        if instance is None:
            return self
        dict_ = instance_dict(instance)
        if self.impl.supports_population and self.key in dict_:
            return dict_[self.key]
        else:
            try:
                state = instance_state(instance)
            except AttributeError as err:
                raise orm_exc.UnmappedInstanceError(instance) from err
            return self.impl.get(state, dict_)
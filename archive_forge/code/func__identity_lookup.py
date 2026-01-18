from __future__ import annotations
import contextlib
from enum import Enum
import itertools
import sys
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import attributes
from . import bulk_persistence
from . import context
from . import descriptor_props
from . import exc
from . import identity
from . import loading
from . import query
from . import state as statelib
from ._typing import _O
from ._typing import insp_is_mapper
from ._typing import is_composite_class
from ._typing import is_orm_option
from ._typing import is_user_defined_option
from .base import _class_to_mapper
from .base import _none_set
from .base import _state_mapper
from .base import instance_str
from .base import LoaderCallableStatus
from .base import object_mapper
from .base import object_state
from .base import PassiveFlag
from .base import state_str
from .context import FromStatement
from .context import ORMCompileState
from .identity import IdentityMap
from .query import Query
from .state import InstanceState
from .state_changes import _StateChange
from .state_changes import _StateChangeState
from .state_changes import _StateChangeStates
from .unitofwork import UOWTransaction
from .. import engine
from .. import exc as sa_exc
from .. import sql
from .. import util
from ..engine import Connection
from ..engine import Engine
from ..engine.util import TransactionalContext
from ..event import dispatcher
from ..event import EventTarget
from ..inspection import inspect
from ..inspection import Inspectable
from ..sql import coercions
from ..sql import dml
from ..sql import roles
from ..sql import Select
from ..sql import TableClause
from ..sql import visitors
from ..sql.base import _NoArg
from ..sql.base import CompileState
from ..sql.schema import Table
from ..sql.selectable import ForUpdateArg
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..util import IdentitySet
from ..util.typing import Literal
from ..util.typing import Protocol
def _identity_lookup(self, mapper: Mapper[_O], primary_key_identity: Union[Any, Tuple[Any, ...]], identity_token: Any=None, passive: PassiveFlag=PassiveFlag.PASSIVE_OFF, lazy_loaded_from: Optional[InstanceState[Any]]=None, execution_options: OrmExecuteOptionsParameter=util.EMPTY_DICT, bind_arguments: Optional[_BindArguments]=None) -> Union[Optional[_O], LoaderCallableStatus]:
    """Locate an object in the identity map.

        Given a primary key identity, constructs an identity key and then
        looks in the session's identity map.  If present, the object may
        be run through unexpiration rules (e.g. load unloaded attributes,
        check if was deleted).

        e.g.::

            obj = session._identity_lookup(inspect(SomeClass), (1, ))

        :param mapper: mapper in use
        :param primary_key_identity: the primary key we are searching for, as
         a tuple.
        :param identity_token: identity token that should be used to create
         the identity key.  Used as is, however overriding subclasses can
         repurpose this in order to interpret the value in a special way,
         such as if None then look among multiple target tokens.
        :param passive: passive load flag passed to
         :func:`.loading.get_from_identity`, which impacts the behavior if
         the object is found; the object may be validated and/or unexpired
         if the flag allows for SQL to be emitted.
        :param lazy_loaded_from: an :class:`.InstanceState` that is
         specifically asking for this identity as a related identity.  Used
         for sharding schemes where there is a correspondence between an object
         and a related object being lazy-loaded (or otherwise
         relationship-loaded).

        :return: None if the object is not found in the identity map, *or*
         if the object was unexpired and found to have been deleted.
         if passive flags disallow SQL and the object is expired, returns
         PASSIVE_NO_RESULT.   In all other cases the instance is returned.

        .. versionchanged:: 1.4.0 - the :meth:`.Session._identity_lookup`
           method was moved from :class:`_query.Query` to
           :class:`.Session`, to avoid having to instantiate the
           :class:`_query.Query` object.


        """
    key = mapper.identity_key_from_primary_key(primary_key_identity, identity_token=identity_token)
    return_value = loading.get_from_identity(self, mapper, key, passive)
    return return_value
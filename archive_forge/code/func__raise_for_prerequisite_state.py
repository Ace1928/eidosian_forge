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
def _raise_for_prerequisite_state(self, operation_name: str, state: _StateChangeState) -> NoReturn:
    if state is SessionTransactionState.DEACTIVE:
        if self._rollback_exception:
            raise sa_exc.PendingRollbackError(f"This Session's transaction has been rolled back due to a previous exception during flush. To begin a new transaction with this Session, first issue Session.rollback(). Original exception was: {self._rollback_exception}", code='7s2a')
        else:
            raise sa_exc.InvalidRequestError("This session is in 'inactive' state, due to the SQL transaction being rolled back; no further SQL can be emitted within this transaction.")
    elif state is SessionTransactionState.CLOSED:
        raise sa_exc.ResourceClosedError('This transaction is closed')
    elif state is SessionTransactionState.PROVISIONING_CONNECTION:
        raise sa_exc.InvalidRequestError('This session is provisioning a new connection; concurrent operations are not permitted', code='isce')
    else:
        raise sa_exc.InvalidRequestError(f"This session is in '{state.name.lower()}' state; no further SQL can be emitted within this transaction.")
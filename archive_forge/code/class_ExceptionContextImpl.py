from __future__ import annotations
import contextlib
import sys
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from .interfaces import BindTyping
from .interfaces import ConnectionEventsTarget
from .interfaces import DBAPICursor
from .interfaces import ExceptionContext
from .interfaces import ExecuteStyle
from .interfaces import ExecutionContext
from .interfaces import IsolationLevel
from .util import _distill_params_20
from .util import _distill_raw_params
from .util import TransactionalContext
from .. import exc
from .. import inspection
from .. import log
from .. import util
from ..sql import compiler
from ..sql import util as sql_util
class ExceptionContextImpl(ExceptionContext):
    """Implement the :class:`.ExceptionContext` interface."""
    __slots__ = ('connection', 'engine', 'dialect', 'cursor', 'statement', 'parameters', 'original_exception', 'sqlalchemy_exception', 'chained_exception', 'execution_context', 'is_disconnect', 'invalidate_pool_on_disconnect', 'is_pre_ping')

    def __init__(self, exception: BaseException, sqlalchemy_exception: Optional[exc.StatementError], engine: Optional[Engine], dialect: Dialect, connection: Optional[Connection], cursor: Optional[DBAPICursor], statement: Optional[str], parameters: Optional[_DBAPIAnyExecuteParams], context: Optional[ExecutionContext], is_disconnect: bool, invalidate_pool_on_disconnect: bool, is_pre_ping: bool):
        self.engine = engine
        self.dialect = dialect
        self.connection = connection
        self.sqlalchemy_exception = sqlalchemy_exception
        self.original_exception = exception
        self.execution_context = context
        self.statement = statement
        self.parameters = parameters
        self.is_disconnect = is_disconnect
        self.invalidate_pool_on_disconnect = invalidate_pool_on_disconnect
        self.is_pre_ping = is_pre_ping
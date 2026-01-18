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
def _execute_context(self, dialect: Dialect, constructor: Callable[..., ExecutionContext], statement: Union[str, Compiled], parameters: Optional[_AnyMultiExecuteParams], execution_options: _ExecuteOptions, *args: Any, **kw: Any) -> CursorResult[Any]:
    """Create an :class:`.ExecutionContext` and execute, returning
        a :class:`_engine.CursorResult`."""
    if execution_options:
        yp = execution_options.get('yield_per', None)
        if yp:
            execution_options = execution_options.union({'stream_results': True, 'max_row_buffer': yp})
    try:
        conn = self._dbapi_connection
        if conn is None:
            conn = self._revalidate_connection()
        context = constructor(dialect, self, conn, execution_options, *args, **kw)
    except (exc.PendingRollbackError, exc.ResourceClosedError):
        raise
    except BaseException as e:
        self._handle_dbapi_exception(e, str(statement), parameters, None, None)
    if self._transaction and (not self._transaction.is_active) or (self._nested_transaction and (not self._nested_transaction.is_active)):
        self._invalid_transaction()
    elif self._trans_context_manager:
        TransactionalContext._trans_ctx_check(self)
    if self._transaction is None:
        self._autobegin()
    context.pre_exec()
    if context.execute_style is ExecuteStyle.INSERTMANYVALUES:
        return self._exec_insertmany_context(dialect, context)
    else:
        return self._exec_single_context(dialect, context, statement, parameters)
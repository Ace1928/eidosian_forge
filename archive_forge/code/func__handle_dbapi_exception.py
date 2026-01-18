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
def _handle_dbapi_exception(self, e: BaseException, statement: Optional[str], parameters: Optional[_AnyExecuteParams], cursor: Optional[DBAPICursor], context: Optional[ExecutionContext], is_sub_exec: bool=False) -> NoReturn:
    exc_info = sys.exc_info()
    is_exit_exception = util.is_exit_exception(e)
    if not self._is_disconnect:
        self._is_disconnect = isinstance(e, self.dialect.loaded_dbapi.Error) and (not self.closed) and self.dialect.is_disconnect(e, self._dbapi_connection if not self.invalidated else None, cursor) or (is_exit_exception and (not self.closed))
    invalidate_pool_on_disconnect = not is_exit_exception
    ismulti: bool = not is_sub_exec and context.executemany if context is not None else False
    if self._reentrant_error:
        raise exc.DBAPIError.instance(statement, parameters, e, self.dialect.loaded_dbapi.Error, hide_parameters=self.engine.hide_parameters, dialect=self.dialect, ismulti=ismulti).with_traceback(exc_info[2]) from e
    self._reentrant_error = True
    try:
        should_wrap = isinstance(e, self.dialect.loaded_dbapi.Error) or (statement is not None and context is None and (not is_exit_exception))
        if should_wrap:
            sqlalchemy_exception = exc.DBAPIError.instance(statement, parameters, cast(Exception, e), self.dialect.loaded_dbapi.Error, hide_parameters=self.engine.hide_parameters, connection_invalidated=self._is_disconnect, dialect=self.dialect, ismulti=ismulti)
        else:
            sqlalchemy_exception = None
        newraise = None
        if self.dialect._has_events and (not self._execution_options.get('skip_user_error_events', False)):
            ctx = ExceptionContextImpl(e, sqlalchemy_exception, self.engine, self.dialect, self, cursor, statement, parameters, context, self._is_disconnect, invalidate_pool_on_disconnect, False)
            for fn in self.dialect.dispatch.handle_error:
                try:
                    per_fn = fn(ctx)
                    if per_fn is not None:
                        ctx.chained_exception = newraise = per_fn
                except Exception as _raised:
                    newraise = _raised
                    break
            if self._is_disconnect != ctx.is_disconnect:
                self._is_disconnect = ctx.is_disconnect
                if sqlalchemy_exception:
                    sqlalchemy_exception.connection_invalidated = ctx.is_disconnect
            invalidate_pool_on_disconnect = ctx.invalidate_pool_on_disconnect
        if should_wrap and context:
            context.handle_dbapi_exception(e)
        if not self._is_disconnect:
            if cursor:
                self._safe_close_cursor(cursor)
            if not self.in_transaction():
                self._rollback_impl()
        if newraise:
            raise newraise.with_traceback(exc_info[2]) from e
        elif should_wrap:
            assert sqlalchemy_exception is not None
            raise sqlalchemy_exception.with_traceback(exc_info[2]) from e
        else:
            assert exc_info[1] is not None
            raise exc_info[1].with_traceback(exc_info[2])
    finally:
        del self._reentrant_error
        if self._is_disconnect:
            del self._is_disconnect
            if not self.invalidated:
                dbapi_conn_wrapper = self._dbapi_connection
                assert dbapi_conn_wrapper is not None
                if invalidate_pool_on_disconnect:
                    self.engine.pool._invalidate(dbapi_conn_wrapper, e)
                self.invalidate(e)
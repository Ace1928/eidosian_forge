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
def _exec_single_context(self, dialect: Dialect, context: ExecutionContext, statement: Union[str, Compiled], parameters: Optional[_AnyMultiExecuteParams]) -> CursorResult[Any]:
    """continue the _execute_context() method for a single DBAPI
        cursor.execute() or cursor.executemany() call.

        """
    if dialect.bind_typing is BindTyping.SETINPUTSIZES:
        generic_setinputsizes = context._prepare_set_input_sizes()
        if generic_setinputsizes:
            try:
                dialect.do_set_input_sizes(context.cursor, generic_setinputsizes, context)
            except BaseException as e:
                self._handle_dbapi_exception(e, str(statement), parameters, None, context)
    cursor, str_statement, parameters = (context.cursor, context.statement, context.parameters)
    effective_parameters: Optional[_AnyExecuteParams]
    if not context.executemany:
        effective_parameters = parameters[0]
    else:
        effective_parameters = parameters
    if self._has_events or self.engine._has_events:
        for fn in self.dispatch.before_cursor_execute:
            str_statement, effective_parameters = fn(self, cursor, str_statement, effective_parameters, context, context.executemany)
    if self._echo:
        self._log_info(str_statement)
        stats = context._get_cache_stats()
        if not self.engine.hide_parameters:
            self._log_info('[%s] %r', stats, sql_util._repr_params(effective_parameters, batches=10, ismulti=context.executemany))
        else:
            self._log_info('[%s] [SQL parameters hidden due to hide_parameters=True]', stats)
    evt_handled: bool = False
    try:
        if context.execute_style is ExecuteStyle.EXECUTEMANY:
            effective_parameters = cast('_CoreMultiExecuteParams', effective_parameters)
            if self.dialect._has_events:
                for fn in self.dialect.dispatch.do_executemany:
                    if fn(cursor, str_statement, effective_parameters, context):
                        evt_handled = True
                        break
            if not evt_handled:
                self.dialect.do_executemany(cursor, str_statement, effective_parameters, context)
        elif not effective_parameters and context.no_parameters:
            if self.dialect._has_events:
                for fn in self.dialect.dispatch.do_execute_no_params:
                    if fn(cursor, str_statement, context):
                        evt_handled = True
                        break
            if not evt_handled:
                self.dialect.do_execute_no_params(cursor, str_statement, context)
        else:
            effective_parameters = cast('_CoreSingleExecuteParams', effective_parameters)
            if self.dialect._has_events:
                for fn in self.dialect.dispatch.do_execute:
                    if fn(cursor, str_statement, effective_parameters, context):
                        evt_handled = True
                        break
            if not evt_handled:
                self.dialect.do_execute(cursor, str_statement, effective_parameters, context)
        if self._has_events or self.engine._has_events:
            self.dispatch.after_cursor_execute(self, cursor, str_statement, effective_parameters, context, context.executemany)
        context.post_exec()
        result = context._setup_result_proxy()
    except BaseException as e:
        self._handle_dbapi_exception(e, str_statement, effective_parameters, cursor, context)
    return result
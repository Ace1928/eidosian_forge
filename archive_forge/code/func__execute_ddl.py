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
def _execute_ddl(self, ddl: ExecutableDDLElement, distilled_parameters: _CoreMultiExecuteParams, execution_options: CoreExecuteOptionsParameter) -> CursorResult[Any]:
    """Execute a schema.DDL object."""
    exec_opts = ddl._execution_options.merge_with(self._execution_options, execution_options)
    event_multiparams: Optional[_CoreMultiExecuteParams]
    event_params: Optional[_CoreSingleExecuteParams]
    if self._has_events or self.engine._has_events:
        ddl, distilled_parameters, event_multiparams, event_params = self._invoke_before_exec_event(ddl, distilled_parameters, exec_opts)
    else:
        event_multiparams = event_params = None
    schema_translate_map = exec_opts.get('schema_translate_map', None)
    dialect = self.dialect
    compiled = ddl.compile(dialect=dialect, schema_translate_map=schema_translate_map)
    ret = self._execute_context(dialect, dialect.execution_ctx_cls._init_ddl, compiled, None, exec_opts, compiled)
    if self._has_events or self.engine._has_events:
        self.dispatch.after_execute(self, ddl, event_multiparams, event_params, exec_opts, ret)
    return ret
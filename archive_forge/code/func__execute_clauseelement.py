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
def _execute_clauseelement(self, elem: Executable, distilled_parameters: _CoreMultiExecuteParams, execution_options: CoreExecuteOptionsParameter) -> CursorResult[Any]:
    """Execute a sql.ClauseElement object."""
    execution_options = elem._execution_options.merge_with(self._execution_options, execution_options)
    has_events = self._has_events or self.engine._has_events
    if has_events:
        elem, distilled_parameters, event_multiparams, event_params = self._invoke_before_exec_event(elem, distilled_parameters, execution_options)
    if distilled_parameters:
        keys = sorted(distilled_parameters[0])
        for_executemany = len(distilled_parameters) > 1
    else:
        keys = []
        for_executemany = False
    dialect = self.dialect
    schema_translate_map = execution_options.get('schema_translate_map', None)
    compiled_cache: Optional[CompiledCacheType] = execution_options.get('compiled_cache', self.engine._compiled_cache)
    compiled_sql, extracted_params, cache_hit = elem._compile_w_cache(dialect=dialect, compiled_cache=compiled_cache, column_keys=keys, for_executemany=for_executemany, schema_translate_map=schema_translate_map, linting=self.dialect.compiler_linting | compiler.WARN_LINTING)
    ret = self._execute_context(dialect, dialect.execution_ctx_cls._init_compiled, compiled_sql, distilled_parameters, execution_options, compiled_sql, distilled_parameters, elem, extracted_params, cache_hit=cache_hit)
    if has_events:
        self.dispatch.after_execute(self, elem, event_multiparams, event_params, execution_options, ret)
    return ret
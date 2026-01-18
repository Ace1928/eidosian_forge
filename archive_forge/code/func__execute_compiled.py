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
def _execute_compiled(self, compiled: Compiled, distilled_parameters: _CoreMultiExecuteParams, execution_options: CoreExecuteOptionsParameter=_EMPTY_EXECUTION_OPTS) -> CursorResult[Any]:
    """Execute a sql.Compiled object.

        TODO: why do we have this?   likely deprecate or remove

        """
    execution_options = compiled.execution_options.merge_with(self._execution_options, execution_options)
    if self._has_events or self.engine._has_events:
        compiled, distilled_parameters, event_multiparams, event_params = self._invoke_before_exec_event(compiled, distilled_parameters, execution_options)
    dialect = self.dialect
    ret = self._execute_context(dialect, dialect.execution_ctx_cls._init_compiled, compiled, distilled_parameters, execution_options, compiled, distilled_parameters, None, None)
    if self._has_events or self.engine._has_events:
        self.dispatch.after_execute(self, compiled, event_multiparams, event_params, execution_options, ret)
    return ret
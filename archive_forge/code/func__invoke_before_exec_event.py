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
def _invoke_before_exec_event(self, elem: Any, distilled_params: _CoreMultiExecuteParams, execution_options: _ExecuteOptions) -> Tuple[Any, _CoreMultiExecuteParams, _CoreMultiExecuteParams, _CoreSingleExecuteParams]:
    event_multiparams: _CoreMultiExecuteParams
    event_params: _CoreSingleExecuteParams
    if len(distilled_params) == 1:
        event_multiparams, event_params = ([], distilled_params[0])
    else:
        event_multiparams, event_params = (distilled_params, {})
    for fn in self.dispatch.before_execute:
        elem, event_multiparams, event_params = fn(self, elem, event_multiparams, event_params, execution_options)
    if event_multiparams:
        distilled_params = list(event_multiparams)
        if event_params:
            raise exc.InvalidRequestError("Event handler can't return non-empty multiparams and params at the same time")
    elif event_params:
        distilled_params = [event_params]
    else:
        distilled_params = []
    return (elem, distilled_params, event_multiparams, event_params)
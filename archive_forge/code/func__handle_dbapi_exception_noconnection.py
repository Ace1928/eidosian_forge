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
@classmethod
def _handle_dbapi_exception_noconnection(cls, e: BaseException, dialect: Dialect, engine: Optional[Engine]=None, is_disconnect: Optional[bool]=None, invalidate_pool_on_disconnect: bool=True, is_pre_ping: bool=False) -> NoReturn:
    exc_info = sys.exc_info()
    if is_disconnect is None:
        is_disconnect = isinstance(e, dialect.loaded_dbapi.Error) and dialect.is_disconnect(e, None, None)
    should_wrap = isinstance(e, dialect.loaded_dbapi.Error)
    if should_wrap:
        sqlalchemy_exception = exc.DBAPIError.instance(None, None, cast(Exception, e), dialect.loaded_dbapi.Error, hide_parameters=engine.hide_parameters if engine is not None else False, connection_invalidated=is_disconnect, dialect=dialect)
    else:
        sqlalchemy_exception = None
    newraise = None
    if dialect._has_events:
        ctx = ExceptionContextImpl(e, sqlalchemy_exception, engine, dialect, None, None, None, None, None, is_disconnect, invalidate_pool_on_disconnect, is_pre_ping)
        for fn in dialect.dispatch.handle_error:
            try:
                per_fn = fn(ctx)
                if per_fn is not None:
                    ctx.chained_exception = newraise = per_fn
            except Exception as _raised:
                newraise = _raised
                break
        if sqlalchemy_exception and is_disconnect != ctx.is_disconnect:
            sqlalchemy_exception.connection_invalidated = is_disconnect = ctx.is_disconnect
    if newraise:
        raise newraise.with_traceback(exc_info[2]) from e
    elif should_wrap:
        assert sqlalchemy_exception is not None
        raise sqlalchemy_exception.with_traceback(exc_info[2]) from e
    else:
        assert exc_info[1] is not None
        raise exc_info[1].with_traceback(exc_info[2])
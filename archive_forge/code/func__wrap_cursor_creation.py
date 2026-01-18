from __future__ import annotations
import contextlib
from typing import Any, TypeVar, Callable, Awaitable, Iterator
from asyncpg.cursor import BaseCursor  # type: ignore
from sentry_sdk import Hub
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.tracing import Span
from sentry_sdk.tracing_utils import add_query_source, record_sql_queries
from sentry_sdk.utils import parse_version, capture_internal_exceptions
def _wrap_cursor_creation(f: Callable[..., T]) -> Callable[..., T]:

    def _inner(*args: Any, **kwargs: Any) -> T:
        hub = Hub.current
        integration = hub.get_integration(AsyncPGIntegration)
        if integration is None:
            return f(*args, **kwargs)
        query = args[1]
        params_list = args[2] if len(args) > 2 else None
        with _record(hub, None, query, params_list, executemany=False) as span:
            _set_db_data(span, args[0])
            res = f(*args, **kwargs)
            span.set_data('db.cursor', res)
        return res
    return _inner
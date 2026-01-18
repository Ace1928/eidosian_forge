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
def _wrap_connect_addr(f: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:

    async def _inner(*args: Any, **kwargs: Any) -> T:
        hub = Hub.current
        integration = hub.get_integration(AsyncPGIntegration)
        if integration is None:
            return await f(*args, **kwargs)
        user = kwargs['params'].user
        database = kwargs['params'].database
        with hub.start_span(op=OP.DB, description='connect') as span:
            span.set_data(SPANDATA.DB_SYSTEM, 'postgresql')
            addr = kwargs.get('addr')
            if addr:
                try:
                    span.set_data(SPANDATA.SERVER_ADDRESS, addr[0])
                    span.set_data(SPANDATA.SERVER_PORT, addr[1])
                except IndexError:
                    pass
            span.set_data(SPANDATA.DB_NAME, database)
            span.set_data(SPANDATA.DB_USER, user)
            with capture_internal_exceptions():
                hub.add_breadcrumb(message='connect', category='query', data=span._data)
            res = await f(*args, **kwargs)
        return res
    return _inner
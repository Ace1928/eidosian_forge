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
class AsyncPGIntegration(Integration):
    identifier = 'asyncpg'
    _record_params = False

    def __init__(self, *, record_params: bool=False):
        AsyncPGIntegration._record_params = record_params

    @staticmethod
    def setup_once() -> None:
        asyncpg.Connection.execute = _wrap_execute(asyncpg.Connection.execute)
        asyncpg.Connection._execute = _wrap_connection_method(asyncpg.Connection._execute)
        asyncpg.Connection._executemany = _wrap_connection_method(asyncpg.Connection._executemany, executemany=True)
        asyncpg.Connection.cursor = _wrap_cursor_creation(asyncpg.Connection.cursor)
        asyncpg.Connection.prepare = _wrap_connection_method(asyncpg.Connection.prepare)
        asyncpg.connect_utils._connect_addr = _wrap_connect_addr(asyncpg.connect_utils._connect_addr)
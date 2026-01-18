from .pymysql import MySQLDialect_pymysql
from ... import pool
from ... import util
from ...engine import AdaptedConnection
from ...util.concurrency import asyncio
from ...util.concurrency import await_fallback
from ...util.concurrency import await_only
class MySQLDialect_aiomysql(MySQLDialect_pymysql):
    driver = 'aiomysql'
    supports_statement_cache = True
    supports_server_side_cursors = True
    _sscursor = AsyncAdapt_aiomysql_ss_cursor
    is_async = True
    has_terminate = True

    @classmethod
    def import_dbapi(cls):
        return AsyncAdapt_aiomysql_dbapi(__import__('aiomysql'), __import__('pymysql'))

    @classmethod
    def get_pool_class(cls, url):
        async_fallback = url.query.get('async_fallback', False)
        if util.asbool(async_fallback):
            return pool.FallbackAsyncAdaptedQueuePool
        else:
            return pool.AsyncAdaptedQueuePool

    def do_terminate(self, dbapi_connection) -> None:
        dbapi_connection.terminate()

    def create_connect_args(self, url):
        return super().create_connect_args(url, _translate_args=dict(username='user', database='db'))

    def is_disconnect(self, e, connection, cursor):
        if super().is_disconnect(e, connection, cursor):
            return True
        else:
            str_e = str(e).lower()
            return 'not connected' in str_e

    def _found_rows_client_flag(self):
        from pymysql.constants import CLIENT
        return CLIENT.FOUND_ROWS

    def get_driver_connection(self, connection):
        return connection._connection
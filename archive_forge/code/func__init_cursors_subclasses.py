from .pymysql import MySQLDialect_pymysql
from ... import pool
from ... import util
from ...engine import AdaptedConnection
from ...util.concurrency import asyncio
from ...util.concurrency import await_fallback
from ...util.concurrency import await_only
def _init_cursors_subclasses(self):

    class Cursor(self.aiomysql.Cursor):

        async def _show_warnings(self, conn):
            pass

    class SSCursor(self.aiomysql.SSCursor):

        async def _show_warnings(self, conn):
            pass
    return (Cursor, SSCursor)
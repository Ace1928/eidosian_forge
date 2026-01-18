from .pymysql import MySQLDialect_pymysql
from ... import pool
from ... import util
from ...engine import AdaptedConnection
from ...util.concurrency import asyncio
from ...util.concurrency import await_fallback
from ...util.concurrency import await_only
class SSCursor(self.aiomysql.SSCursor):

    async def _show_warnings(self, conn):
        pass
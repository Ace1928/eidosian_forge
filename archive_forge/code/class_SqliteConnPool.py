import asyncio
import sqlite3
import json
import logging
import aiosqlite
from .defaults import SQLITE_THREADS, SQLITE_TIMEOUT
from .base_cache import BaseCache, CacheEntry
class SqliteConnPool:

    def __init__(self, threads, conn_args=(), conn_kwargs=None, init_queries=()):
        self._threads = threads
        self._conn_args = conn_args
        self._conn_kwargs = conn_kwargs if conn_kwargs is not None else {}
        self._init_queries = init_queries
        self._free_conns = asyncio.Queue()
        self._ready = False
        self._stopped = False

    async def _new_conn(self):
        db = await aiosqlite.connect(*self._conn_args, **self._conn_kwargs)
        try:
            async with db.cursor() as cur:
                for q in self._init_queries:
                    await cur.execute(q)
        except:
            await db.close()
            raise
        return db

    async def prepare(self):
        for _ in range(self._threads):
            self._free_conns.put_nowait(await self._new_conn())
        self._ready = True

    async def stop(self):
        self._ready = False
        self._stopped = True
        try:
            while True:
                db = self._free_conns.get_nowait()
                await db.close()
        except asyncio.QueueEmpty:
            pass

    def borrow(self, timeout=None):
        if not self._ready:
            raise RuntimeError('Pool not prepared!')

        class PoolBorrow:

            def __init__(s):
                s._conn = None

            async def __aenter__(s):
                s._conn = await asyncio.wait_for(self._free_conns.get(), timeout)
                return s._conn

            async def __aexit__(s, exc_type, exc, tb):
                if self._stopped:
                    await s._conn.close()
                    return
                if exc_type is not None:
                    await s._conn.close()
                    s._conn = await self._new_conn()
                self._free_conns.put_nowait(s._conn)
        return PoolBorrow()
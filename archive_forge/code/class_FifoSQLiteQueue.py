import glob
import json
import os
import sqlite3
import struct
from abc import abstractmethod
from collections import deque
from contextlib import suppress
from typing import Any, Optional
class FifoSQLiteQueue:
    _sql_create = 'CREATE TABLE IF NOT EXISTS queue (id INTEGER PRIMARY KEY AUTOINCREMENT, item BLOB)'
    _sql_size = 'SELECT COUNT(*) FROM queue'
    _sql_push = 'INSERT INTO queue (item) VALUES (?)'
    _sql_pop = 'SELECT id, item FROM queue ORDER BY id LIMIT 1'
    _sql_del = 'DELETE FROM queue WHERE id = ?'

    def __init__(self, path: str) -> None:
        self._path = os.path.abspath(path)
        self._db = sqlite3.Connection(self._path, timeout=60)
        self._db.text_factory = bytes
        with self._db as conn:
            conn.execute(self._sql_create)

    def push(self, item: bytes) -> None:
        if not isinstance(item, bytes):
            raise TypeError('Unsupported type: {}'.format(type(item).__name__))
        with self._db as conn:
            conn.execute(self._sql_push, (item,))

    def pop(self) -> Optional[bytes]:
        with self._db as conn:
            for id_, item in conn.execute(self._sql_pop):
                conn.execute(self._sql_del, (id_,))
                return item
        return None

    def peek(self) -> Optional[bytes]:
        with self._db as conn:
            for _, item in conn.execute(self._sql_pop):
                return item
        return None

    def close(self) -> None:
        size = len(self)
        self._db.close()
        if not size:
            os.remove(self._path)

    def __len__(self) -> int:
        with self._db as conn:
            return next(conn.execute(self._sql_size))[0]
import asyncio
import logging
import sqlite3
from functools import partial
from pathlib import Path
from queue import Empty, Queue, SimpleQueue
from threading import Thread
from typing import (
from warnings import warn
from .context import contextmanager
from .cursor import Cursor
def _execute_fetchall(self, sql: str, parameters: Any) -> Iterable[sqlite3.Row]:
    cursor = self._conn.execute(sql, parameters)
    return cursor.fetchall()
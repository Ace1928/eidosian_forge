from __future__ import annotations
import collections
import re
import typing
from typing import Any
from typing import Dict
from typing import Optional
import warnings
import weakref
from . import config
from .util import decorator
from .util import gc_collect
from .. import event
from .. import pool
from ..util import await_only
from ..util.typing import Literal
class DBAPIProxyCursor:
    """Proxy a DBAPI cursor.

    Tests can provide subclasses of this to intercept
    DBAPI-level cursor operations.

    """

    def __init__(self, engine, conn, *args, **kwargs):
        self.engine = engine
        self.connection = conn
        self.cursor = conn.cursor(*args, **kwargs)

    def execute(self, stmt, parameters=None, **kw):
        if parameters:
            return self.cursor.execute(stmt, parameters, **kw)
        else:
            return self.cursor.execute(stmt, **kw)

    def executemany(self, stmt, params, **kw):
        return self.cursor.executemany(stmt, params, **kw)

    def __iter__(self):
        return iter(self.cursor)

    def __getattr__(self, key):
        return getattr(self.cursor, key)
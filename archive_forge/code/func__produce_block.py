import contextlib
import functools
import inspect
import operator
import threading
import warnings
import debtcollector.moves
import debtcollector.removals
import debtcollector.renames
from oslo_config import cfg
from oslo_utils import excutils
from oslo_db import exception
from oslo_db import options
from oslo_db.sqlalchemy import engines
from oslo_db.sqlalchemy import orm
from oslo_db import warning
def _produce_block(self, mode, connection, savepoint, allow_async=False, context=None):
    if mode is _WRITER:
        self._writer()
    elif mode is _ASYNC_READER:
        self._async_reader()
    else:
        self._reader(allow_async)
    if connection:
        return self._connection(savepoint, context=context)
    else:
        return self._session(savepoint, context=context)
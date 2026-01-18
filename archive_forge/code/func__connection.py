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
@contextlib.contextmanager
def _connection(self, savepoint=False, context=None):
    if self.connection is None:
        try:
            if self.session is not None:
                self.connection = self.session.connection()
                if savepoint:
                    with self.connection.begin_nested(), self._add_context(self.connection, context):
                        yield self.connection
                else:
                    with self._add_context(self.connection, context):
                        yield self.connection
            else:
                self.connection = self.factory._create_connection(mode=self.mode)
                self.transaction = self.connection.begin()
                try:
                    with self._add_context(self.connection, context):
                        yield self.connection
                    self._end_connection_transaction(self.transaction)
                except Exception:
                    self.transaction.rollback()
                    raise
                finally:
                    self.transaction = None
                    self.connection.close()
        finally:
            self.connection = None
    elif savepoint:
        with self.connection.begin_nested(), self._add_context(self.connection, context):
            yield self.connection
    else:
        with self._add_context(self.connection, context):
            yield self.connection
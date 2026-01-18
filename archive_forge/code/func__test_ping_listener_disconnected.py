import contextlib
import itertools
from unittest import mock
import sqlalchemy as sqla
from sqlalchemy import event
import sqlalchemy.exc
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import registry
from sqlalchemy import sql
from oslo_db import exception
from oslo_db.sqlalchemy import compat
from oslo_db.sqlalchemy import engines
from oslo_db.sqlalchemy import exc_filters
from oslo_db.sqlalchemy import utils
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
from oslo_db.tests import utils as test_utils
def _test_ping_listener_disconnected(self, dialect_name, exc_obj, is_disconnect=True):
    with self._fixture(dialect_name, exc_obj, 3, is_disconnect):
        conn = self.engine.connect()
        self.assertEqual(1, conn.scalar(sqla.select(1)))
        conn.close()
    with self._fixture(dialect_name, exc_obj, 1, is_disconnect):
        self.assertRaises(exception.DBConnectionError, self.engine.connect)
        self.assertRaises(exception.DBConnectionError, self.engine.connect)
        self.assertRaises(exception.DBConnectionError, self.engine.connect)
    with self._fixture(dialect_name, exc_obj, 1, is_disconnect):
        self.assertRaises(exception.DBConnectionError, self.engine.connect)
        self.assertRaises(exception.DBConnectionError, self.engine.connect)
        self.assertRaises(exception.DBConnectionError, self.engine.connect)
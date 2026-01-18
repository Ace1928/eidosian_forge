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
class TestDBConnectRetry(TestsExceptionFilter):

    def _run_test(self, dialect_name, exception, count, retries):
        counter = itertools.count()
        engine = self.engine
        engine.dispose()
        connect_fn = engine.dialect.connect

        def cant_connect(*arg, **kw):
            if next(counter) < count:
                raise exception
            else:
                return connect_fn(*arg, **kw)
        with self._dbapi_fixture(dialect_name):
            with mock.patch.object(engine.dialect, 'connect', cant_connect):
                return engines._test_connection(engine, retries, 0.01)

    def test_connect_no_retries(self):
        conn = self._run_test('mysql', self.OperationalError('Error: (2003) something wrong'), 2, 0)
        self.assertIsNone(conn)

    def test_connect_inifinite_retries(self):
        conn = self._run_test('mysql', self.OperationalError('Error: (2003) something wrong'), 2, -1)
        self.assertEqual(1, conn.scalar(sqla.select(1)))

    def test_connect_retry_past_failure(self):
        conn = self._run_test('mysql', self.OperationalError('Error: (2003) something wrong'), 2, 3)
        self.assertEqual(1, conn.scalar(sqla.select(1)))

    def test_connect_retry_not_candidate_exception(self):
        self.assertRaises(sqla.exc.OperationalError, self._run_test, 'mysql', self.OperationalError("Error: (2015) I can't connect period"), 2, 3)

    def test_connect_retry_stops_infailure(self):
        self.assertRaises(exception.DBConnectionError, self._run_test, 'mysql', self.OperationalError('Error: (2003) something wrong'), 3, 2)
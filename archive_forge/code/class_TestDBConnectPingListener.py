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
class TestDBConnectPingListener(TestsExceptionFilter):

    def setUp(self):
        super().setUp()
        event.listen(self.engine, 'engine_connect', engines._connect_ping_listener)

    @contextlib.contextmanager
    def _fixture(self, dialect_name, exception, good_conn_count, is_disconnect=True):
        engine = self.engine
        engine.dispose()
        connect_fn = engine.dialect.connect
        real_do_execute = engine.dialect.do_execute
        counter = itertools.count(1)

        def cant_execute(*arg, **kw):
            value = next(counter)
            if value > good_conn_count:
                raise exception
            else:
                return real_do_execute(*arg, **kw)

        def cant_connect(*arg, **kw):
            value = next(counter)
            if value > good_conn_count:
                raise exception
            else:
                return connect_fn(*arg, **kw)
        with self._dbapi_fixture(dialect_name, is_disconnect=is_disconnect):
            with mock.patch.object(engine.dialect, 'connect', cant_connect):
                with mock.patch.object(engine.dialect, 'do_execute', cant_execute):
                    yield

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

    def test_mysql_w_disconnect_flag(self):
        for code in [2002, 2003, 2002]:
            self._test_ping_listener_disconnected('mysql', self.OperationalError('%d MySQL server has gone away' % code))

    def test_mysql_wo_disconnect_flag(self):
        for code in [2002, 2003]:
            self._test_ping_listener_disconnected('mysql', self.OperationalError('%d MySQL server has gone away' % code), is_disconnect=False)
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
class TestDBDisconnectedFixture(TestsExceptionFilter):
    native_pre_ping = False

    def _test_ping_listener_disconnected(self, dialect_name, exc_obj, is_disconnect=True):
        with self._fixture(dialect_name, exc_obj, False, is_disconnect) as engine:
            conn = engine.connect()
            with conn.begin():
                self.assertEqual(1, conn.execute(sqla.select(1)).scalars().first())
                self.assertFalse(conn.closed)
                self.assertFalse(conn.invalidated)
                self.assertTrue(conn.in_transaction())
        with self._fixture(dialect_name, exc_obj, True, is_disconnect) as engine:
            self.assertRaises(exception.DBConnectionError, engine.connect)
        with self._fixture(dialect_name, exc_obj, False) as engine:
            with engine.connect() as conn:
                self.assertEqual(1, conn.execute(sqla.select(1)).scalars().first())

    @contextlib.contextmanager
    def _fixture(self, dialect_name, exception, db_stays_down, is_disconnect=True):
        """Fixture for testing the ping listener.

        For SQLAlchemy 2.0, the mocking is placed more deeply in the
        stack within the DBAPI connection / cursor so that we can also
        effectively mock out the "pre ping" condition.

        :param dialect_name: dialect to use.  "postgresql" or "mysql"
        :param exception: an exception class to raise
        :param db_stays_down: if True, the database will stay down after the
         first ping fails
        :param is_disconnect: whether or not the SQLAlchemy dialect should
         consider the exception object as a "disconnect error".   Openstack's
         own exception handlers upgrade various DB exceptions to be
         "disconnect" scenarios that SQLAlchemy itself does not, such as
         some specific Galera error messages.

        The importance of an exception being a "disconnect error" means that
        SQLAlchemy knows it can discard the connection and then reconnect.
        If the error is not a "disconnection error", then it raises.
        """
        connect_args = {}
        patchers = []
        db_disconnected = False

        class DisconnectCursorMixin:

            def execute(self, *arg, **kw):
                if db_disconnected:
                    raise exception
                else:
                    return super().execute(*arg, **kw)
        if dialect_name == 'postgresql':
            import psycopg2.extensions

            class Curs(DisconnectCursorMixin, psycopg2.extensions.cursor):
                pass
            connect_args = {'cursor_factory': Curs}
        elif dialect_name == 'mysql':
            import pymysql

            def fake_ping(self, *arg, **kw):
                if db_disconnected:
                    raise exception
                else:
                    return True

            class Curs(DisconnectCursorMixin, pymysql.cursors.Cursor):
                pass
            connect_args = {'cursorclass': Curs}
            patchers.append(mock.patch.object(pymysql.Connection, 'ping', fake_ping))
        else:
            raise NotImplementedError()
        with mock.patch.object(compat, 'native_pre_ping_event_support', self.native_pre_ping):
            engine = engines.create_engine(self.engine.url, max_retries=0)

        @event.listens_for(engine, 'do_connect')
        def _connect(dialect, connrec, cargs, cparams):
            nonlocal db_disconnected
            cparams.update(connect_args)
            if db_disconnected:
                if db_stays_down:
                    raise exception
                else:
                    db_disconnected = False
        conn = engine.connect()
        conn.close()
        patchers.extend([mock.patch.object(engine.dialect.dbapi, 'Error', self.Error), mock.patch.object(engine.dialect, 'is_disconnect', mock.Mock(return_value=is_disconnect))])
        with test_utils.nested(*patchers):
            db_disconnected = True
            yield engine
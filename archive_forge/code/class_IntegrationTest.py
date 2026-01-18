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
class IntegrationTest(db_test_base._DbTestCase):
    """Test an actual error-raising round trips against the database."""

    def setUp(self):
        super(IntegrationTest, self).setUp()
        meta = sqla.MetaData()
        self.test_table = sqla.Table(_TABLE_NAME, meta, sqla.Column('id', sqla.Integer, primary_key=True, nullable=False), sqla.Column('counter', sqla.Integer, nullable=False), sqla.UniqueConstraint('counter', name='uniq_counter'))
        self.test_table.create(self.engine)
        self.addCleanup(self.test_table.drop, self.engine)
        reg = registry()

        class Foo(object):

            def __init__(self, counter):
                self.counter = counter
        reg.map_imperatively(Foo, self.test_table)
        self.Foo = Foo

    def test_flush_wrapper_duplicate_entry(self):
        """test a duplicate entry exception."""
        _session = self.sessionmaker()
        with _session.begin():
            foo = self.Foo(counter=1)
            _session.add(foo)
        _session.begin()
        self.addCleanup(_session.rollback)
        foo = self.Foo(counter=1)
        _session.add(foo)
        self.assertRaises(exception.DBDuplicateEntry, _session.flush)

    def test_autoflush_wrapper_duplicate_entry(self):
        """Test a duplicate entry exception raised.

        test a duplicate entry exception raised via query.all()-> autoflush
        """
        _session = self.sessionmaker()
        with _session.begin():
            foo = self.Foo(counter=1)
            _session.add(foo)
        _session.begin()
        self.addCleanup(_session.rollback)
        foo = self.Foo(counter=1)
        _session.add(foo)
        self.assertTrue(_session.autoflush)
        self.assertRaises(exception.DBDuplicateEntry, _session.query(self.Foo).all)

    def test_flush_wrapper_plain_integrity_error(self):
        """test a plain integrity error wrapped as DBError."""
        _session = self.sessionmaker()
        with _session.begin():
            foo = self.Foo(counter=1)
            _session.add(foo)
        _session.begin()
        self.addCleanup(_session.rollback)
        foo = self.Foo(counter=None)
        _session.add(foo)
        self.assertRaises(exception.DBError, _session.flush)

    def test_flush_wrapper_operational_error(self):
        """test an operational error from flush() raised as-is."""
        _session = self.sessionmaker()
        with _session.begin():
            foo = self.Foo(counter=1)
            _session.add(foo)
        _session.begin()
        self.addCleanup(_session.rollback)
        foo = self.Foo(counter=sqla.func.imfake(123))
        _session.add(foo)
        matched = self.assertRaises(sqla.exc.OperationalError, _session.flush)
        self.assertIn('no such function', str(matched))

    def test_query_wrapper_operational_error(self):
        """test an operational error from query.all() raised as-is."""
        _session = self.sessionmaker()
        _session.begin()
        self.addCleanup(_session.rollback)
        q = _session.query(self.Foo).filter(self.Foo.counter == sqla.func.imfake(123))
        matched = self.assertRaises(sqla.exc.OperationalError, q.all)
        self.assertIn('no such function', str(matched))
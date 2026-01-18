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
class TestReferenceErrorSQLite(_SQLAExceptionMatcher, db_test_base._DbTestCase):

    def setUp(self):
        super(TestReferenceErrorSQLite, self).setUp()
        meta = sqla.MetaData()
        self.table_1 = sqla.Table('resource_foo', meta, sqla.Column('id', sqla.Integer, primary_key=True), sqla.Column('foo', sqla.Integer), mysql_engine='InnoDB', mysql_charset='utf8')
        self.table_1.create(self.engine)
        self.table_2 = sqla.Table('resource_entity', meta, sqla.Column('id', sqla.Integer, primary_key=True), sqla.Column('foo_id', sqla.Integer, sqla.ForeignKey('resource_foo.id', name='foo_fkey')), mysql_engine='InnoDB', mysql_charset='utf8')
        self.table_2.create(self.engine)

    def test_raise(self):
        connection = self.engine.raw_connection()
        try:
            cursor = connection.cursor()
            cursor.execute('PRAGMA foreign_keys = ON')
            cursor.close()
        finally:
            connection.close()
        with self.engine.connect() as conn:
            matched = self.assertRaises(exception.DBReferenceError, conn.execute, self.table_2.insert().values(id=1, foo_id=2))
        self.assertInnerException(matched, 'IntegrityError', 'FOREIGN KEY constraint failed', 'INSERT INTO resource_entity (id, foo_id) VALUES (?, ?)', (1, 2))
        self.assertIsNone(matched.table)
        self.assertIsNone(matched.constraint)
        self.assertIsNone(matched.key)
        self.assertIsNone(matched.key_table)

    def test_raise_delete(self):
        connection = self.engine.raw_connection()
        try:
            cursor = connection.cursor()
            cursor.execute('PRAGMA foreign_keys = ON')
            cursor.close()
        finally:
            connection.close()
        with self.engine.connect() as conn:
            with conn.begin():
                conn.execute(self.table_1.insert().values(id=1234, foo=42))
                conn.execute(self.table_2.insert().values(id=4321, foo_id=1234))
                matched = self.assertRaises(exception.DBReferenceError, conn.execute, self.table_1.delete())
        self.assertInnerException(matched, 'IntegrityError', 'foreign key constraint failed', 'DELETE FROM resource_foo', ())
        self.assertIsNone(matched.table)
        self.assertIsNone(matched.constraint)
        self.assertIsNone(matched.key)
        self.assertIsNone(matched.key_table)
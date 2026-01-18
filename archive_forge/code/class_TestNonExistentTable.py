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
class TestNonExistentTable(_SQLAExceptionMatcher, db_test_base._DbTestCase):

    def setUp(self):
        super(TestNonExistentTable, self).setUp()
        self.meta = sqla.MetaData()
        self.table_1 = sqla.Table('foo', self.meta, sqla.Column('id', sqla.Integer, primary_key=True), mysql_engine='InnoDB', mysql_charset='utf8')

    def test_raise(self):
        with self.engine.connect() as conn:
            matched = self.assertRaises(exception.DBNonExistentTable, conn.execute, sqla.schema.DropTable(self.table_1))
        self.assertInnerException(matched, 'OperationalError', 'no such table: foo', '\nDROP TABLE foo')
        self.assertEqual('foo', matched.table)
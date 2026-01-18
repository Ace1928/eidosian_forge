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
class TestDuplicate(TestsExceptionFilter):

    def _run_dupe_constraint_test(self, dialect_name, message, expected_columns=['a', 'b'], expected_value=None):
        matched = self._run_test(dialect_name, 'insert into table some_values', self.IntegrityError(message), exception.DBDuplicateEntry)
        self.assertEqual(expected_columns, matched.columns)
        self.assertEqual(expected_value, matched.value)

    def _not_dupe_constraint_test(self, dialect_name, statement, message, expected_cls):
        matched = self._run_test(dialect_name, statement, self.IntegrityError(message), expected_cls)
        self.assertInnerException(matched, 'IntegrityError', str(self.IntegrityError(message)), statement)

    def test_sqlite(self):
        self._run_dupe_constraint_test('sqlite', 'column a, b are not unique')

    def test_sqlite_3_7_16_or_3_8_2_and_higher(self):
        self._run_dupe_constraint_test('sqlite', 'UNIQUE constraint failed: tbl.a, tbl.b')

    def test_sqlite_dupe_primary_key(self):
        self._run_dupe_constraint_test('sqlite', "PRIMARY KEY must be unique 'insert into t values(10)'", expected_columns=[])

    def test_mysql_pymysql(self):
        self._run_dupe_constraint_test('mysql', '(1062, "Duplicate entry \'2-3\' for key \'uniq_tbl0a0b\'")', expected_value='2-3')
        self._run_dupe_constraint_test('mysql', '(1062, "Duplicate entry \'\' for key \'uniq_tbl0a0b\'")', expected_value='')

    def test_mysql_mysqlconnector(self):
        self._run_dupe_constraint_test('mysql', '1062 (23000): Duplicate entry \'2-3\' for key \'uniq_tbl0a0b\'")', expected_value='2-3')

    def test_postgresql(self):
        self._run_dupe_constraint_test('postgresql', 'duplicate key value violates unique constraint"uniq_tbl0a0b"\nDETAIL:  Key (a, b)=(2, 3) already exists.\n', expected_value='2, 3')

    def test_mysql_single(self):
        self._run_dupe_constraint_test('mysql', "1062 (23000): Duplicate entry '2' for key 'b'", expected_columns=['b'], expected_value='2')

    def test_mysql_duplicate_entry_key_start_with_tablename(self):
        self._run_dupe_constraint_test('mysql', "1062 (23000): Duplicate entry '2' for key 'tbl.uniq_tbl0b'", expected_columns=['b'], expected_value='2')

    def test_mysql_binary(self):
        self._run_dupe_constraint_test('mysql', '(1062, \'Duplicate entry \\\'\\\\x8A$\\\\x8D\\\\xA6"s\\\\x8E\\\' for key \\\'PRIMARY\\\'\')', expected_columns=['PRIMARY'], expected_value='\\\\x8A$\\\\x8D\\\\xA6"s\\\\x8E')
        self._run_dupe_constraint_test('mysql', '(1062, \'Duplicate entry \'\'\\\\x8A$\\\\x8D\\\\xA6"s\\\\x8E!,\' for key \'PRIMARY\'\')', expected_columns=['PRIMARY'], expected_value='\'\\\\x8A$\\\\x8D\\\\xA6"s\\\\x8E!,')

    def test_mysql_duplicate_entry_key_start_with_tablename_binary(self):
        self._run_dupe_constraint_test('mysql', '(1062, \'Duplicate entry \\\'\\\\x8A$\\\\x8D\\\\xA6"s\\\\x8E\\\' for key \\\'tbl.uniq_tbl0c1\\\'\')', expected_columns=['c1'], expected_value='\\\\x8A$\\\\x8D\\\\xA6"s\\\\x8E')
        self._run_dupe_constraint_test('mysql', '(1062, \'Duplicate entry \'\'\\\\x8A$\\\\x8D\\\\xA6"s\\\\x8E!,\' for key \'tbl.uniq_tbl0c1\'\')', expected_columns=['c1'], expected_value='\'\\\\x8A$\\\\x8D\\\\xA6"s\\\\x8E!,')

    def test_postgresql_single(self):
        self._run_dupe_constraint_test('postgresql', 'duplicate key value violates unique constraint "uniq_tbl0b"\nDETAIL:  Key (b)=(2) already exists.\n', expected_columns=['b'], expected_value='2')

    def test_unsupported_backend(self):
        self._not_dupe_constraint_test('nonexistent', 'insert into table some_values', self.IntegrityError('constraint violation'), exception.DBError)
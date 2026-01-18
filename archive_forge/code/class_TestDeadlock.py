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
class TestDeadlock(TestsExceptionFilter):
    statement = 'SELECT quota_usages.created_at AS quota_usages_created_at FROM quota_usages WHERE quota_usages.project_id = :project_id_1 AND quota_usages.deleted = :deleted_1 FOR UPDATE'
    params = {'project_id_1': '8891d4478bbf48ad992f050cdf55e9b5', 'deleted_1': 0}

    def _run_deadlock_detect_test(self, dialect_name, message, orig_exception_cls=TestsExceptionFilter.OperationalError):
        self._run_test(dialect_name, self.statement, orig_exception_cls(message), exception.DBDeadlock, params=self.params)

    def _not_deadlock_test(self, dialect_name, message, expected_cls, expected_dbapi_cls, orig_exception_cls=TestsExceptionFilter.OperationalError):
        matched = self._run_test(dialect_name, self.statement, orig_exception_cls(message), expected_cls, params=self.params)
        if isinstance(matched, exception.DBError):
            matched = matched.inner_exception
        self.assertEqual(expected_dbapi_cls, matched.orig.__class__.__name__)

    def test_mysql_pymysql_deadlock(self):
        self._run_deadlock_detect_test('mysql', "(1213, 'Deadlock found when trying to get lock; try restarting transaction')")

    def test_mysql_pymysql_wsrep_deadlock(self):
        self._run_deadlock_detect_test('mysql', "(1213, 'WSREP detected deadlock/conflict and aborted the transaction. Try restarting the transaction')", orig_exception_cls=self.InternalError)
        self._run_deadlock_detect_test('mysql', "(1213, 'Deadlock: wsrep aborted transaction')", orig_exception_cls=self.InternalError)

    def test_mysql_pymysql_galera_deadlock(self):
        self._run_deadlock_detect_test('mysql', "(1205, 'Lock wait timeout exceeded; try restarting transaction')", orig_exception_cls=self.InternalError)

    def test_mysql_mysqlconnector_deadlock(self):
        self._run_deadlock_detect_test('mysql', '1213 (40001): Deadlock found when trying to get lock; try restarting transaction', orig_exception_cls=self.InternalError)

    def test_mysql_not_deadlock(self):
        self._not_deadlock_test('mysql', "(1005, 'some other error')", sqla.exc.OperationalError, 'OperationalError')

    def test_postgresql_deadlock(self):
        self._run_deadlock_detect_test('postgresql', 'deadlock detected', orig_exception_cls=self.TransactionRollbackError)

    def test_postgresql_not_deadlock(self):
        self._not_deadlock_test('postgresql', 'relation "fake" does not exist', (exception.DBError, sqla.exc.OperationalError), 'TransactionRollbackError', orig_exception_cls=self.TransactionRollbackError)
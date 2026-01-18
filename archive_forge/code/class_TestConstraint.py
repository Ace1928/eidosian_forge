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
class TestConstraint(TestsExceptionFilter):

    def test_postgresql(self):
        matched = self._run_test('postgresql', 'insert into resource some_values', self.IntegrityError('new row for relation "resource" violates check constraint "ck_started_before_ended"'), exception.DBConstraintError)
        self.assertEqual('resource', matched.table)
        self.assertEqual('ck_started_before_ended', matched.check_name)
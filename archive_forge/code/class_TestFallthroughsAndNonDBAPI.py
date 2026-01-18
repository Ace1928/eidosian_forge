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
class TestFallthroughsAndNonDBAPI(TestsExceptionFilter):

    def test_generic_dbapi(self):
        matched = self._run_test('mysql', 'select you_made_a_programming_error', self.ProgrammingError('Error 123, you made a mistake'), exception.DBError)
        self.assertInnerException(matched, 'ProgrammingError', 'Error 123, you made a mistake', 'select you_made_a_programming_error', ())

    def test_generic_dbapi_disconnect(self):
        matched = self._run_test('mysql', 'select the_db_disconnected', self.InterfaceError('connection lost'), exception.DBConnectionError, is_disconnect=True)
        (self.assertInnerException(matched, 'InterfaceError', 'connection lost', 'select the_db_disconnected', ()),)

    def test_operational_dbapi_disconnect(self):
        matched = self._run_test('mysql', 'select the_db_disconnected', self.OperationalError('connection lost'), exception.DBConnectionError, is_disconnect=True)
        (self.assertInnerException(matched, 'OperationalError', 'connection lost', 'select the_db_disconnected', ()),)

    def test_operational_error_asis(self):
        """Test operational errors.

        test that SQLAlchemy OperationalErrors that aren't disconnects
        are passed through without wrapping.
        """
        matched = self._run_test('mysql', 'select some_operational_error', self.OperationalError('some op error'), sqla.exc.OperationalError)
        self.assertSQLAException(matched, 'OperationalError', 'some op error')

    def test_unicode_encode(self):
        uee_ref = None
        try:
            '\u2435'.encode('ascii')
        except UnicodeEncodeError as uee:
            uee_ref = uee
        self._run_test('postgresql', 'select \u2435', uee_ref, exception.DBInvalidUnicodeParameter)

    def test_garden_variety(self):
        matched = self._run_test('mysql', 'select some_thing_that_breaks', AttributeError('mysqldb has an attribute error'), exception.DBError)
        self.assertEqual('mysqldb has an attribute error', matched.args[0])
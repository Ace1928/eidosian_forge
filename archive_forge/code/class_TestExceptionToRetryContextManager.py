from unittest import mock
from oslo_db import exception as db_exc
import osprofiler
import sqlalchemy
from sqlalchemy.orm import exc
import testtools
from neutron_lib.db import api as db_api
from neutron_lib import exceptions
from neutron_lib import fixture
from neutron_lib.tests import _base
class TestExceptionToRetryContextManager(_base.BaseTestCase):

    def test_translates_single_exception(self):
        with testtools.ExpectedException(db_exc.RetryRequest):
            with db_api.exc_to_retry(ValueError):
                raise ValueError()

    def test_translates_multiple_exception_types(self):
        with testtools.ExpectedException(db_exc.RetryRequest):
            with db_api.exc_to_retry((ValueError, TypeError)):
                raise TypeError()

    def test_translates_DBerror_inner_exception(self):
        with testtools.ExpectedException(db_exc.RetryRequest):
            with db_api.exc_to_retry(ValueError):
                raise db_exc.DBError(ValueError())

    def test_passes_other_exceptions(self):
        with testtools.ExpectedException(ValueError):
            with db_api.exc_to_retry(TypeError):
                raise ValueError()

    def test_inner_exception_preserved_in_retryrequest(self):
        try:
            exc = ValueError('test')
            with db_api.exc_to_retry(ValueError):
                raise exc
        except db_exc.RetryRequest as e:
            self.assertEqual(exc, e.inner_exc)

    def test_retries_on_multi_exception_containing_target(self):
        with testtools.ExpectedException(db_exc.RetryRequest):
            with db_api.exc_to_retry(ValueError):
                e = exceptions.MultipleExceptions([ValueError(), TypeError()])
                raise e
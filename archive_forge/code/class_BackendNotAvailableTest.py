import os
from unittest import mock
from sqlalchemy.engine import url as sqla_url
from sqlalchemy import exc as sa_exc
from sqlalchemy import inspect
from sqlalchemy import schema
from sqlalchemy import types
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import test_fixtures
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
class BackendNotAvailableTest(test_base.BaseTestCase):

    def test_no_dbapi(self):
        backend = provision.Backend('postgresql', 'postgresql+nosuchdbapi://hostname/dsn')
        with mock.patch('sqlalchemy.create_engine', mock.Mock(side_effect=ImportError('nosuchdbapi'))):
            ex = self.assertRaises(exception.BackendNotAvailable, backend._verify)
            self.assertEqual("Backend 'postgresql+nosuchdbapi' is unavailable: No DBAPI installed", str(ex))
            ex = self.assertRaises(exception.BackendNotAvailable, backend._verify)
            self.assertEqual("Backend 'postgresql+nosuchdbapi' is unavailable: No DBAPI installed", str(ex))

    def test_cant_connect(self):
        backend = provision.Backend('postgresql', 'postgresql+nosuchdbapi://hostname/dsn')
        with mock.patch('sqlalchemy.create_engine', mock.Mock(return_value=mock.Mock(connect=mock.Mock(side_effect=sa_exc.OperationalError("can't connect", None, None))))):
            ex = self.assertRaises(exception.BackendNotAvailable, backend._verify)
            self.assertEqual("Backend 'postgresql+nosuchdbapi' is unavailable: Could not connect", str(ex))
            ex = self.assertRaises(exception.BackendNotAvailable, backend._verify)
            self.assertEqual("Backend 'postgresql+nosuchdbapi' is unavailable: Could not connect", str(ex))
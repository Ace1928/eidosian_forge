import sys
from unittest import mock
from oslo_config import fixture as config_fixture
from oslo_db import concurrency
from oslo_db.tests import base as test_base
class TpoolDbapiWrapperTestCase(test_base.BaseTestCase):

    def setUp(self):
        super(TpoolDbapiWrapperTestCase, self).setUp()
        self.conf = self.useFixture(config_fixture.Config()).conf
        self.db_api = concurrency.TpoolDbapiWrapper(conf=self.conf, backend_mapping=FAKE_BACKEND_MAPPING)
        self.proxy = mock.MagicMock()
        self.eventlet = mock.MagicMock()
        self.eventlet.tpool.Proxy.return_value = self.proxy
        sys.modules['eventlet'] = self.eventlet
        self.addCleanup(sys.modules.pop, 'eventlet', None)

    @mock.patch('oslo_db.api.DBAPI')
    def test_db_api_common(self, mock_db_api):
        fake_db_api = mock.MagicMock()
        mock_db_api.from_config.return_value = fake_db_api
        self.db_api.fake_call_1
        mock_db_api.from_config.assert_called_once_with(conf=self.conf, backend_mapping=FAKE_BACKEND_MAPPING)
        self.assertEqual(fake_db_api, self.db_api._db_api)
        self.assertFalse(self.eventlet.tpool.Proxy.called)
        self.db_api.fake_call_2
        self.assertEqual(fake_db_api, self.db_api._db_api)
        self.assertFalse(self.eventlet.tpool.Proxy.called)
        self.assertEqual(1, mock_db_api.from_config.call_count)

    @mock.patch('oslo_db.api.DBAPI')
    def test_db_api_config_change(self, mock_db_api):
        fake_db_api = mock.MagicMock()
        mock_db_api.from_config.return_value = fake_db_api
        self.conf.set_override('use_tpool', True, group='database')
        self.db_api.fake_call
        mock_db_api.from_config.assert_called_once_with(conf=self.conf, backend_mapping=FAKE_BACKEND_MAPPING)
        self.eventlet.tpool.Proxy.assert_called_once_with(fake_db_api)
        self.assertEqual(self.proxy, self.db_api._db_api)

    @mock.patch('oslo_db.api.DBAPI')
    def test_db_api_without_installed_eventlet(self, mock_db_api):
        self.conf.set_override('use_tpool', True, group='database')
        sys.modules['eventlet'] = None
        self.assertRaises(ImportError, getattr, self.db_api, 'fake')
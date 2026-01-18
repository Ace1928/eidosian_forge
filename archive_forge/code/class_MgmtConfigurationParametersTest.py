import json
import testtools
from unittest import mock
from troveclient import base
from troveclient.v1 import configurations
from troveclient.v1 import management
class MgmtConfigurationParametersTest(testtools.TestCase):

    def setUp(self):
        super(MgmtConfigurationParametersTest, self).setUp()
        self.orig__init = management.MgmtConfigurationParameters.__init__
        management.MgmtConfigurationParameters.__init__ = mock.Mock(return_value=None)
        self.config_params = management.MgmtConfigurationParameters()
        self.config_params.api = mock.Mock()
        self.config_params.api.client = mock.Mock()

    def tearDown(self):
        super(MgmtConfigurationParametersTest, self).tearDown()
        management.MgmtConfigurationParameters.__init__ = self.orig__init

    def _get_mock_method(self):
        self._resp = mock.Mock()
        self._body = None
        self._url = None

        def side_effect_func(url, body=None):
            self._body = body
            self._url = url
            return (self._resp, body)
        return mock.Mock(side_effect=side_effect_func)

    def test_create(self):
        self.config_params.api.client.post = self._get_mock_method()
        self._resp.status_code = 200
        self.config_params.create('id', 'config_name', 1, 'string')
        self.assertEqual('/mgmt/datastores/versions/id/parameters', self._url)
        expected = {'name': 'config_name', 'data_type': 'string', 'restart_required': 1}
        self.assertEqual({'configuration-parameter': expected}, self._body)

    def test_modify(self):
        self.config_params.api.client.put = self._get_mock_method()
        self._resp.status_code = 200
        self.config_params.modify('id', 'config_name', '1', 'string')
        self.assertEqual('/mgmt/datastores/versions/id/parameters/config_name', self._url)
        expected = {'name': 'config_name', 'data_type': 'string', 'restart_required': 1}
        self.assertEqual({'configuration-parameter': expected}, self._body)

    def test_delete(self):
        self.config_params.api.client.delete = self._get_mock_method()
        self._resp.status_code = 200
        self.config_params.delete('id', 'param_id')
        self.assertEqual('/mgmt/datastores/versions/id/parameters/param_id', self._url)
        self.assertIsNone(self._body)
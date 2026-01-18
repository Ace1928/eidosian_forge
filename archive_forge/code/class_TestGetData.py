from unittest import mock
from keystoneauth1 import adapter
from openstack.baremetal.v1 import node as _node
from openstack.baremetal_introspection.v1 import _proxy
from openstack.baremetal_introspection.v1 import introspection
from openstack.baremetal_introspection.v1 import introspection_rule
from openstack import exceptions
from openstack.tests.unit import base
from openstack.tests.unit import test_proxy_base
@mock.patch.object(_proxy.Proxy, 'request', autospec=True)
class TestGetData(base.TestCase):

    def setUp(self):
        super(TestGetData, self).setUp()
        self.session = mock.Mock(spec=adapter.Adapter)
        self.proxy = _proxy.Proxy(self.session)
        self.fake = {'id': '1234', 'finished': False}
        self.introspection = introspection.Introspection(**self.fake)

    def test_get_data(self, mock_request):
        mock_request.return_value.status_code = 200
        data = self.proxy.get_introspection_data(self.introspection)
        mock_request.assert_called_once_with(self.proxy, 'introspection/1234/data', 'GET', headers=mock.ANY, microversion=mock.ANY)
        self.assertIs(data, mock_request.return_value.json.return_value)

    def test_get_unprocessed_data(self, mock_request):
        mock_request.return_value.status_code = 200
        data = self.proxy.get_introspection_data(self.introspection, processed=False)
        mock_request.assert_called_once_with(self.proxy, 'introspection/1234/data/unprocessed', 'GET', headers=mock.ANY, microversion='1.17')
        self.assertIs(data, mock_request.return_value.json.return_value)
from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from keystoneauth1 import loading as keystone
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
from neutron_lib._i18n import _
from neutron_lib.exceptions import placement as n_exc
from neutron_lib import fixture
from neutron_lib.placement import client as place_client
from neutron_lib.tests import _base as base
class TestNoAuthClient(base.BaseTestCase):

    def setUp(self):
        super(TestNoAuthClient, self).setUp()
        self.noauth_client = place_client.NoAuthClient('placement/')
        self.body_json = jsonutils.dumps({'name': 'foo'})
        self.uuid = '42'

    @mock.patch.object(place_client.NoAuthClient, 'request')
    def test_get(self, mock_request):
        self.noauth_client.get('resource_providers', '')
        mock_request.assert_called_with('placement/resource_providers', 'GET')

    @mock.patch.object(place_client.NoAuthClient, 'request')
    def test_post(self, mock_request):
        self.noauth_client.post('resource_providers', self.body_json, '')
        mock_request.assert_called_with('placement/resource_providers', 'POST', body=self.body_json)

    @mock.patch.object(place_client.NoAuthClient, 'request')
    def test_put(self, mock_request):
        self.noauth_client.put('resource_providers/%s' % self.uuid, self.body_json, '')
        mock_request.assert_called_with('placement/resource_providers/%s' % self.uuid, 'PUT', body=self.body_json)

    @mock.patch.object(place_client.NoAuthClient, 'request')
    def test_delete(self, mock_request):
        self.noauth_client.delete('resource_providers/%s' % self.uuid, '')
        mock_request.assert_called_with('placement/resource_providers/%s' % self.uuid, 'DELETE')
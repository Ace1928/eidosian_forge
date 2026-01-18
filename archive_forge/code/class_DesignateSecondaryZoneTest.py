from unittest import mock
from heat.common import exception
from heat.engine.resources.openstack.designate import zone
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
class DesignateSecondaryZoneTest(common.HeatTestCase):

    def setUp(self):
        super(DesignateSecondaryZoneTest, self).setUp()
        self.ctx = utils.dummy_context()
        self.primaries = ['::1']

    def _get_mock_resource(self):
        value = {}
        value['id'] = '477e8273-60a7-4c41-b683-fdb0bc7cd152'
        value['serial'] = '1434596972'
        return value

    def test_masters(self):
        self.do_test({'heat_template_version': '2015-04-30', 'resources': {'test_resource': {'type': 'OS::Designate::Zone', 'properties': {'name': 'test-zone.com', 'description': 'Test zone', 'ttl': 3600, 'email': 'abc@test-zone.com', 'type': 'SECONDARY', 'masters': self.primaries}}}})

    def test_primaries(self):
        self.do_test({'heat_template_version': '2015-04-30', 'resources': {'test_resource': {'type': 'OS::Designate::Zone', 'properties': {'name': 'test-zone.com', 'description': 'Test zone', 'ttl': 3600, 'email': 'abc@test-zone.com', 'type': 'SECONDARY', 'primaries': self.primaries}}}})

    def do_test(self, sample_template):
        self.stack = stack.Stack(self.ctx, 'test_stack', template.Template(sample_template))
        self.test_resource = self.stack['test_resource']
        self.test_client_plugin = mock.MagicMock()
        self.test_resource.client_plugin = mock.MagicMock(return_value=self.test_client_plugin)
        self.test_client = mock.MagicMock()
        self.test_resource.client = mock.MagicMock(return_value=self.test_client)
        mock_zone_create = self.test_client.zones.create
        mock_resource = self._get_mock_resource()
        mock_zone_create.return_value = mock_resource
        self.test_resource.data_set = mock.Mock()
        self.test_resource.handle_create()
        args = dict(name='test-zone.com', description='Test zone', ttl=3600, email='abc@test-zone.com', type_='SECONDARY', masters=self.primaries)
        mock_zone_create.assert_called_once_with(**args)
        self.assertEqual(mock_resource['id'], self.test_resource.resource_id)
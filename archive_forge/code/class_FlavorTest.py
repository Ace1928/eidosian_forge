from unittest import mock
from heat.common import template_format
from heat.tests import common
from heat.tests.openstack.octavia import inline_templates
from heat.tests import utils
class FlavorTest(common.HeatTestCase):

    def _create_stack(self, tmpl=inline_templates.FLAVOR_TEMPLATE):
        self.t = template_format.parse(tmpl)
        self.stack = utils.parse_stack(self.t)
        self.flavor = self.stack['flavor']
        self.octavia_client = mock.MagicMock()
        self.flavor.client = mock.MagicMock()
        self.flavor.client.return_value = self.octavia_client
        self.flavor.client_plugin().client = mock.MagicMock(return_value=self.octavia_client)
        self.patchobject(self.flavor, 'physical_resource_name', return_value='resource_name')

    def test_create(self):
        self._create_stack()
        self.octavia_client.flavor_show.side_effect = [{'flavor': {'id': 'f123'}}]
        expected = {'flavor': {'name': 'test_name', 'description': 'test_description', 'flavor_profile_id': 'test_flavor_profile_id', 'enabled': True}}
        self.flavor.handle_create()
        self.octavia_client.flavor_create.assert_called_with(json=expected)

    def test_update(self):
        self._create_stack()
        self.flavor.resource_id_set('f123')
        prop_diff = {'name': 'test_name2', 'description': 'test_description2', 'flavor_profile_id': 'test_flavor_profile_id2', 'enabled': False}
        self.flavor.handle_update(None, None, prop_diff)
        self.octavia_client.flavor_set.assert_called_once_with('f123', json={'flavor': prop_diff})
        self.octavia_client.flavor_set.reset_mock()
        prop_diff = {'name': None, 'description': 'test_description3', 'flavor_profile_id': 'test_flavor_profile_id3', 'enabled': True}
        self.flavor.handle_update(None, None, prop_diff)
        self.assertEqual(prop_diff['name'], 'resource_name')
        self.octavia_client.flavor_set.assert_called_once_with('f123', json={'flavor': prop_diff})

    def test_delete(self):
        self._create_stack()
        self.flavor.resource_id_set('f123')
        self.flavor.handle_delete()
        self.octavia_client.flavor_delete.assert_called_with('f123')
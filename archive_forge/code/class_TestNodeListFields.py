from tempest.lib import exceptions
from ironicclient.tests.functional.osc.v1 import base
class TestNodeListFields(base.TestCase):
    """Functional tests for "baremetal node list" with --fields."""

    def setUp(self):
        super(TestNodeListFields, self).setUp()
        self.node = self.node_create()

    def _get_table_headers(self, raw_output):
        table = self.parser.table(raw_output)
        return table['headers']

    def test_list_default_fields(self):
        """Test presence of default list table headers."""
        headers = ['UUID', 'Name', 'Instance UUID', 'Power State', 'Provisioning State', 'Maintenance']
        nodes_list = self.openstack('baremetal node list')
        nodes_list_headers = self._get_table_headers(nodes_list)
        self.assertEqual(set(headers), set(nodes_list_headers))

    def test_list_minimal_fields(self):
        headers = ['Instance UUID', 'Name', 'UUID']
        fields = ['instance_uuid', 'name', 'uuid']
        node_list = self.openstack('baremetal node list --fields {}'.format(' '.join(fields)))
        nodes_list_headers = self._get_table_headers(node_list)
        self.assertEqual(headers, nodes_list_headers)

    def test_list_no_fields(self):
        command = 'baremetal node list --fields'
        ex_text = 'expected at least one argument'
        self.assertRaisesRegex(exceptions.CommandFailed, ex_text, self.openstack, command)

    def test_list_wrong_field(self):
        command = 'baremetal node list --fields ABC'
        ex_text = 'invalid choice'
        self.assertRaisesRegex(exceptions.CommandFailed, ex_text, self.openstack, command)
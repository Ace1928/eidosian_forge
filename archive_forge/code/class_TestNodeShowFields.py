from tempest.lib import exceptions
from ironicclient.tests.functional.osc.v1 import base
class TestNodeShowFields(base.TestCase):
    """Functional tests for "baremetal node show" with --fields."""

    def setUp(self):
        super(TestNodeShowFields, self).setUp()
        self.node = self.node_create()
        self.api_version = '--os-baremetal-api-version 1.20'

    def _get_table_rows(self, raw_output):
        table = self.parser.table(raw_output)
        rows = []
        for row in table['values']:
            rows.append(row[0])
        return rows

    def test_show_default_fields(self):
        rows = ['console_enabled', 'clean_step', 'created_at', 'deploy_step', 'driver', 'driver_info', 'driver_internal_info', 'extra', 'inspection_finished_at', 'inspection_started_at', 'instance_info', 'instance_uuid', 'last_error', 'maintenance', 'maintenance_reason', 'name', 'power_state', 'properties', 'provision_state', 'provision_updated_at', 'reservation', 'target_power_state', 'target_provision_state', 'updated_at', 'uuid']
        node_show = self.openstack('baremetal node show {}'.format(self.node['uuid']))
        nodes_show_rows = self._get_table_rows(node_show)
        self.assertTrue(set(rows).issubset(set(nodes_show_rows)))

    def test_show_minimal_fields(self):
        rows = ['instance_uuid', 'name', 'uuid']
        node_show = self.openstack('baremetal node show {} --fields {} {}'.format(self.node['uuid'], ' '.join(rows), self.api_version))
        nodes_show_rows = self._get_table_rows(node_show)
        self.assertEqual(set(rows), set(nodes_show_rows))

    def test_show_no_fields(self):
        command = 'baremetal node show {} --fields {}'.format(self.node['uuid'], self.api_version)
        ex_text = 'expected at least one argument'
        self.assertRaisesRegex(exceptions.CommandFailed, ex_text, self.openstack, command)

    def test_show_wrong_field(self):
        command = 'baremetal node show {} --fields ABC {}'.format(self.node['uuid'], self.api_version)
        ex_text = 'invalid choice'
        self.assertRaisesRegex(exceptions.CommandFailed, ex_text, self.openstack, command)
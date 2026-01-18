import datetime
from novaclient.tests.functional import base
def _get_num_servers_by_tenant_from_usage_output(self):
    tenant_id = self._get_project_id(self.cli_clients.tenant_name)
    output = self.nova('usage --tenant=%s' % tenant_id)
    servers = self._get_column_value_from_single_row_table(output, 'Servers')
    return int(servers)
from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _construct_opcommand_hst(self, operation):
    """Construct operation command host.

        Args:
            operation: operation to construct command host

        Returns:
            list: constructed operation command host
        """
    if operation.get('run_on_hosts') is None:
        return None
    return [{'hostid': self._zapi_wrapper.get_host_by_host_name(_host)['hostid']} if str(_host) != '0' else {'hostid': '0'} for _host in operation.get('run_on_hosts')]
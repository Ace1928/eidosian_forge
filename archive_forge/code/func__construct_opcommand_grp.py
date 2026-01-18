from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _construct_opcommand_grp(self, operation):
    """Construct operation command group.

        Args:
            operation: operation to construct command group

        Returns:
            list: constructed operation command group
        """
    if operation.get('run_on_groups') is None:
        return None
    return [{'groupid': self._zapi_wrapper.get_hostgroup_by_hostgroup_name(_group)['groupid']} for _group in operation.get('run_on_groups')]
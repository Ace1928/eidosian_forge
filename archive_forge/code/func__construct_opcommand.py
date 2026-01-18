from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _construct_opcommand(self, operation):
    """Construct operation command.

        Args:
            operation: operation to construct command

        Returns:
            list: constructed operation command
        """
    try:
        opcommand = {'scriptid': self._zapi_wrapper.get_script_by_script_name(operation.get('script_name')).get('scriptid')}
        return opcommand
    except Exception as e:
        self._module.fail_json(msg='Failed to construct operation command. The error was: %s' % e)
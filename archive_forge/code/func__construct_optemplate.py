from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _construct_optemplate(self, operation):
    """Construct operation template.

        Args:
            operation: operation to construct template

        Returns:
            list: constructed operation template
        """
    return [{'templateid': self._zapi_wrapper.get_template_by_template_name(_template)['templateid']} for _template in operation.get('templates', [])]
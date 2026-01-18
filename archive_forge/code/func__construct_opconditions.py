from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _construct_opconditions(self, operation):
    """Construct operation conditions.

        Args:
            operation: operation to construct the conditions

        Returns:
            list: constructed operation conditions
        """
    _opcond = operation.get('operation_condition')
    if _opcond is not None:
        if _opcond == 'acknowledged':
            _value = '1'
        elif _opcond == 'not_acknowledged':
            _value = '0'
        return [{'conditiontype': '14', 'operator': '0', 'value': _value}]
    return []
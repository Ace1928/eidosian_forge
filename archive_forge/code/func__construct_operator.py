from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _construct_operator(self, _condition):
    """Construct operator

        Args:
            _condition: condition to construct

        Returns:
            str: constructed operator
        """
    try:
        return zabbix_utils.helper_to_numeric_value([['equals', '='], ['does not equal', '<>'], ['contains', 'like'], ['does not contain', 'not like'], 'in', ['is greater than or equals', '>='], ['is less than or equals', '<='], 'not in', 'matches', 'does not match', 'Yes', 'No'], _condition['operator'])
    except Exception:
        self._module.fail_json(msg="Unsupported value '%s' for operator." % _condition['operator'])
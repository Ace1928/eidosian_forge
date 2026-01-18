from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
class RecoveryOperations(Operations):
    """
    Restructures the user defined recovery operations data to fit the Zabbix API requirements
    """

    def _construct_operationtype(self, operation):
        """Construct operation type.

        Args:
            operation: operation to construct type

        Returns:
            str: constructed operation type
        """
        try:
            return zabbix_utils.helper_to_numeric_value(['send_message', 'remote_command', None, None, None, None, None, None, None, None, None, 'notify_all_involved'], operation['type'])
        except Exception:
            self._module.fail_json(msg="Unsupported value '%s' for recovery operation type." % operation['type'])

    def construct_the_data(self, operations):
        """Construct the recovery operations data using helper methods.

        Args:
            operation: operation to construct

        Returns:
            list: constructed recovery operations data
        """
        constructed_data = []
        for op in operations:
            operation_type = self._construct_operationtype(op)
            constructed_operation = {'operationtype': operation_type}
            if constructed_operation['operationtype'] == 0:
                constructed_operation['opmessage'] = self._construct_opmessage(op)
                constructed_operation['opmessage_usr'] = self._construct_opmessage_usr(op)
                constructed_operation['opmessage_grp'] = self._construct_opmessage_grp(op)
            if constructed_operation['operationtype'] == 11:
                constructed_operation['opmessage'] = self._construct_opmessage(op)
                constructed_operation['opmessage'].pop('mediatypeid')
            if constructed_operation['operationtype'] == 1:
                constructed_operation['opcommand'] = self._construct_opcommand(op)
                constructed_operation['opcommand_hst'] = self._construct_opcommand_hst(op)
                constructed_operation['opcommand_grp'] = self._construct_opcommand_grp(op)
            constructed_data.append(constructed_operation)
        return zabbix_utils.helper_cleanup_data(constructed_data)
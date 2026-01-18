from __future__ import absolute_import, division, print_function
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell import (
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.powerflex_base \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
from ansible.module_utils.basic import AnsibleModule
class FaultSetDeleteHandler:

    def handle(self, fault_set_obj, fault_set_params, fault_set_details):
        if fault_set_params['state'] == 'absent' and fault_set_details:
            fault_set_details = fault_set_obj.delete_fault_set(fault_set_details['id'])
            fault_set_obj.result['changed'] = True
        FaultSetExitHandler().handle(fault_set_obj, fault_set_details)
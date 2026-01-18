from __future__ import absolute_import, division, print_function
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell import (
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.powerflex_base \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
from ansible.module_utils.basic import AnsibleModule
class FaultSetCreateHandler:

    def handle(self, fault_set_obj, fault_set_params, fault_set_details, pd_id):
        if fault_set_params['state'] == 'present' and (not fault_set_details):
            fault_set_details = fault_set_obj.create_fault_set(fault_set_name=fault_set_params['fault_set_name'], protection_domain_id=pd_id)
            fault_set_obj.result['changed'] = True
        FaultSetRenameHandler().handle(fault_set_obj, fault_set_params, fault_set_details)
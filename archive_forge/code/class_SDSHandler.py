from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.powerflex_base \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import copy
class SDSHandler:

    def handle(self, sds_obj, sds_params):
        sds_details = sds_obj.get_sds_details(sds_params['sds_name'], sds_params['sds_id'])
        sds_obj.validate_parameters(sds_params=sds_params)
        protection_domain_id = None
        if sds_params['protection_domain_id'] or sds_params['protection_domain_name']:
            protection_domain_id = sds_obj.get_protection_domain(protection_domain_id=sds_params['protection_domain_id'], protection_domain_name=sds_params['protection_domain_name'])['id']
        fault_set_id = None
        if sds_params['fault_set_name'] or sds_params['fault_set_id']:
            fault_set_details = sds_obj.get_fault_set(fault_set_name=sds_params['fault_set_name'], fault_set_id=sds_params['fault_set_id'], protection_domain_id=protection_domain_id)
            if fault_set_details is None:
                error_msg = 'The specified Fault set is not in the specified Protection Domain.'
                LOG.error(error_msg)
                sds_obj.module.fail_json(msg=error_msg)
            else:
                fault_set_id = fault_set_details['id']
        SDSCreateHandler().handle(sds_obj, sds_params, sds_details, protection_domain_id, fault_set_id)
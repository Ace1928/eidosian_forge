from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.powerflex_base \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import copy
class SDSCreateHandler:

    def handle(self, sds_obj, sds_params, sds_details, protection_domain_id, fault_set_id):
        create_flag = False
        sds_ip_list = copy.deepcopy(sds_params['sds_ip_list'])
        if sds_params['state'] == 'present' and (not sds_details):
            sds_details = sds_obj.create_sds(sds_name=sds_params['sds_name'], sds_id=sds_params['sds_id'], sds_new_name=sds_params['sds_new_name'], protection_domain_id=protection_domain_id, sds_ip_list=sds_ip_list, sds_ip_state=sds_params['sds_ip_state'], rmcache_enabled=sds_params['rmcache_enabled'], rmcache_size=sds_params['rmcache_size'], fault_set_id=fault_set_id)
            sds_obj.result['changed'] = True
            create_flag = True
        SDSModifyHandler().handle(sds_obj, sds_params, sds_details, create_flag, sds_ip_list)
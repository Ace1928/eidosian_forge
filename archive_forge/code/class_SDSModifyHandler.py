from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.powerflex_base \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import copy
class SDSModifyHandler:

    def handle(self, sds_obj, sds_params, sds_details, create_flag, sds_ip_list):
        if sds_params['state'] == 'present' and sds_details:
            modify_dict = sds_obj.to_modify(sds_details=sds_details, sds_new_name=sds_params['sds_new_name'], rfcache_enabled=sds_params['rfcache_enabled'], rmcache_enabled=sds_params['rmcache_enabled'], rmcache_size=sds_params['rmcache_size'], performance_profile=sds_params['performance_profile'])
            if modify_dict:
                sds_details = sds_obj.modify_sds_attributes(sds_id=sds_details['id'], modify_dict=modify_dict, create_flag=create_flag)
                sds_obj.result['changed'] = True
        SDSAddIPHandler().handle(sds_obj, sds_params, sds_details, sds_ip_list)
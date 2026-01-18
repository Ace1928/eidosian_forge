from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.powerflex_base \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import copy
class SDSAddIPHandler:

    def handle(self, sds_obj, sds_params, sds_details, sds_ip_list):
        if sds_params['state'] == 'present' and sds_details:
            add_ip_changed = False
            update_role_changed = False
            if sds_params['sds_ip_state'] == 'present-in-sds':
                sds_obj.validate_ip_parameter(sds_ip_list)
                ips_to_add, roles_to_update = sds_obj.identify_ip_role_add(sds_ip_list, sds_details, sds_params['sds_ip_state'])
                if ips_to_add:
                    add_ip_changed = sds_obj.add_ip(sds_details['id'], ips_to_add)
                if roles_to_update:
                    update_role_changed = sds_obj.update_role(sds_details['id'], roles_to_update)
            if add_ip_changed or update_role_changed:
                sds_obj.result['changed'] = True
        SDSRemoveIPHandler().handle(sds_obj, sds_params, sds_details, sds_ip_list)
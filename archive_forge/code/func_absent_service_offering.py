from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def absent_service_offering(self):
    service_offering = self.get_service_offering()
    if service_offering:
        self.result['changed'] = True
        if not self.module.check_mode:
            args = {'id': service_offering['id']}
            self.query_api('deleteServiceOffering', **args)
    return service_offering
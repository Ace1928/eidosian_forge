from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def absent_disk_offering(self):
    disk_offering = self.get_disk_offering()
    if disk_offering:
        self.result['changed'] = True
        if not self.module.check_mode:
            args = {'id': disk_offering['id']}
            self.query_api('deleteDiskOffering', **args)
    return disk_offering
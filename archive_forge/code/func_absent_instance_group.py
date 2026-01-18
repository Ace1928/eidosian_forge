from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def absent_instance_group(self):
    instance_group = self.get_instance_group()
    if instance_group:
        self.result['changed'] = True
        if not self.module.check_mode:
            self.query_api('deleteInstanceGroup', id=instance_group['id'])
    return instance_group
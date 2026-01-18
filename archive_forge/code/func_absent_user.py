from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def absent_user(self):
    user = self.get_user()
    if user:
        self.result['changed'] = True
        if not self.module.check_mode:
            self.query_api('deleteUser', id=user['id'])
    return user
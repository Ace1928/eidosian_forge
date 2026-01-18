from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def _validated_params(self, opt):
    value = self.get_playbook_params(opt)
    if value is None:
        msg = 'Please provide %s option in the playbook!' % opt
        self.module.fail_json(msg=msg)
    return value
from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def _update_role(self, role):
    args = {'id': role['id'], 'name': self.module.params.get('name'), 'description': self.module.params.get('description')}
    if self.has_changed(args, role):
        self.result['changed'] = True
        if not self.module.check_mode:
            res = self.query_api('updateRole', **args)
            if 'role' not in res:
                role = self.get_role()
            else:
                role = res['role']
    return role
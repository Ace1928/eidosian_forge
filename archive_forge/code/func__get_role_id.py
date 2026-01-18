from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import raise_from
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def _get_role_id(self):
    role = self.module.params.get('role')
    if not role:
        return None
    res = self.query_api('listRoles')
    roles = res['role']
    if roles:
        for r in roles:
            if role in [r['name'], r['id']]:
                return r['id']
    self.fail_json(msg="Role '%s' not found" % role)
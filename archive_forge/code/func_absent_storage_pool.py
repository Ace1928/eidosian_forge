from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def absent_storage_pool(self):
    pool = self.get_storage_pool()
    if pool:
        self.result['changed'] = True
        args = {'id': pool['id']}
        if not self.module.check_mode:
            self._handle_allocation_state(pool=pool, state='maintenance')
            self.query_api('deleteStoragePool', **args)
    return pool
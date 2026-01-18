from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def _enable_maintenance(self, pool):
    if pool['state'].lower() != 'maintenance':
        self.result['changed'] = True
        if not self.module.check_mode:
            res = self.query_api('enableStorageMaintenance', id=pool['id'])
            pool = self.poll_job(res, 'storagepool')
    return pool
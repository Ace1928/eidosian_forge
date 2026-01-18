from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def _create_region(self, region):
    self.result['changed'] = True
    args = {'id': self.module.params.get('id'), 'name': self.module.params.get('name'), 'endpoint': self.module.params.get('endpoint')}
    if not self.module.check_mode:
        res = self.query_api('addRegion', **args)
        region = res['region']
    return region
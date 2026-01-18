from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def absent_region(self):
    region = self.get_region()
    if region:
        self.result['changed'] = True
        if not self.module.check_mode:
            self.query_api('removeRegion', id=region['id'])
    return region
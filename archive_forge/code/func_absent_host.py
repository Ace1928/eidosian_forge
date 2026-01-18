from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def absent_host(self):
    host = self.get_host()
    if host:
        self.result['changed'] = True
        args = {'id': host['id']}
        if not self.module.check_mode:
            res = self.enable_maintenance(host)
            if res:
                res = self.query_api('deleteHost', **args)
    return host
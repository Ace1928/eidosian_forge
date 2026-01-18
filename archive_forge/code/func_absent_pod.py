from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def absent_pod(self):
    pod = self.get_pod()
    if pod:
        self.result['changed'] = True
        args = {'id': pod['id']}
        if not self.module.check_mode:
            self.query_api('deletePod', **args)
    return pod
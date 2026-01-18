from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def _update_host(self, host, allocation_state=None):
    args = {'id': host['id'], 'hosttags': self.get_host_tags(), 'allocationstate': allocation_state}
    if allocation_state is not None:
        host = self._set_host_allocation_state(host)
    if self.has_changed(args, host):
        self.result['changed'] = True
        if not self.module.check_mode:
            host = self.query_api('updateHost', **args)
            host = host['host']
    return host
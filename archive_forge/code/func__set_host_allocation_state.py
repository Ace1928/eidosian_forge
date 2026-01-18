from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def _set_host_allocation_state(self, host):
    if host is None:
        host['allocationstate'] = 'Enable'
    elif host['resourcestate'].lower() in list(self.allocation_states_for_update.keys()):
        host['allocationstate'] = self.allocation_states_for_update[host['resourcestate'].lower()]
    else:
        host['allocationstate'] = host['resourcestate']
    return host
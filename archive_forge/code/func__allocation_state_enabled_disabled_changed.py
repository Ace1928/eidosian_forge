from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def _allocation_state_enabled_disabled_changed(self, pool, allocation_state):
    if allocation_state in ['enabled', 'disabled']:
        for pool_state, param_state in self.allocation_states.items():
            if pool_state == pool['state'] and allocation_state != param_state:
                return True
    return False
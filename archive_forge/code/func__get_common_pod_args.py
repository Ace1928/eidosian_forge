from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def _get_common_pod_args(self):
    args = {'name': self.module.params.get('name'), 'zoneid': self.get_zone(key='id'), 'startip': self.module.params.get('start_ip'), 'endip': self.module.params.get('end_ip'), 'netmask': self.module.params.get('netmask'), 'gateway': self.module.params.get('gateway')}
    state = self.module.params.get('state')
    if state in ['enabled', 'disabled']:
        args['allocationstate'] = state.capitalize()
    return args
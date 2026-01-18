from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def absent_nic_ip(self):
    nic = self.get_nic()
    secondary_ip = self.get_secondary_ip()
    if secondary_ip:
        self.result['changed'] = True
        if not self.module.check_mode:
            res = self.query_api('removeIpFromNic', id=secondary_ip['id'])
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                self.poll_job(res, 'nicsecondaryip')
    return nic
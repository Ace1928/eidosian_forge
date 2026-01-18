from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def absent_vpc(self):
    vpc = self.get_vpc()
    if vpc:
        self.result['changed'] = True
        self.result['diff']['before'] = vpc
        if not self.module.check_mode:
            res = self.query_api('deleteVPC', id=vpc['id'])
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                self.poll_job(res, 'vpc')
    return vpc
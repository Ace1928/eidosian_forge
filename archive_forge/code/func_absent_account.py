from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def absent_account(self):
    account = self.get_account()
    if account:
        self.result['changed'] = True
        if not self.module.check_mode:
            res = self.query_api('deleteAccount', id=account['id'])
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                self.poll_job(res, 'account')
    return account
from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def absent_snapshot_policy(self):
    policy = self.get_snapshot_policy()
    if policy:
        self.result['changed'] = True
        args = {'id': policy['id']}
        if not self.module.check_mode:
            self.query_api('deleteSnapshotPolicies', **args)
    return policy
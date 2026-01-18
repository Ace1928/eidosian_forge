from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def _update_lb_rule(self, rule):
    args = {'id': rule['id'], 'algorithm': self.module.params.get('algorithm'), 'description': self.module.params.get('description')}
    if self.has_changed(args, rule):
        self.result['changed'] = True
        if not self.module.check_mode:
            res = self.query_api('updateLoadBalancerRule', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                rule = self.poll_job(res, 'loadbalancer')
    return rule
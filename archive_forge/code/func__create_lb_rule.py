from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def _create_lb_rule(self, rule):
    self.result['changed'] = True
    if not self.module.check_mode:
        args = self._get_common_args()
        args.update({'algorithm': self.module.params.get('algorithm'), 'privateport': self.module.params.get('private_port'), 'publicport': self.module.params.get('public_port'), 'cidrlist': self.module.params.get('cidr'), 'description': self.module.params.get('description'), 'protocol': self.module.params.get('protocol'), 'networkid': self.get_network(key='id')})
        res = self.query_api('createLoadBalancerRule', **args)
        poll_async = self.module.params.get('poll_async')
        if poll_async:
            rule = self.poll_job(res, 'loadbalancer')
    return rule
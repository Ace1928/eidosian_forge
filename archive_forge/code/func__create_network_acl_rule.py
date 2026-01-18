from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def _create_network_acl_rule(self, network_acl_rule):
    self.result['changed'] = True
    protocol = self.module.params.get('protocol')
    args = {'aclid': self.get_network_acl(key='id'), 'action': self.module.params.get('action_policy'), 'protocol': protocol if protocol != 'by_number' else self.module.params.get('protocol_number'), 'startport': self.module.params.get('start_port'), 'endport': self.get_or_fallback('end_port', 'start_port'), 'number': self.module.params.get('rule_position'), 'icmpcode': self.module.params.get('icmp_code'), 'icmptype': self.module.params.get('icmp_type'), 'traffictype': self.module.params.get('traffic_type'), 'cidrlist': self.module.params.get('cidrs')}
    if not self.module.check_mode:
        res = self.query_api('createNetworkACL', **args)
        poll_async = self.module.params.get('poll_async')
        if poll_async:
            network_acl_rule = self.poll_job(res, 'networkacl')
    return network_acl_rule
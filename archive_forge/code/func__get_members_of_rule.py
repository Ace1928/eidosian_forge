from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def _get_members_of_rule(self, rule):
    res = self.query_api('listLoadBalancerRuleInstances', id=rule['id'])
    return res.get('loadbalancerruleinstance', [])
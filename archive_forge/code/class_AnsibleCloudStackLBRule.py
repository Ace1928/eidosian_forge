from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
class AnsibleCloudStackLBRule(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackLBRule, self).__init__(module)
        self.returns = {'publicip': 'public_ip', 'algorithm': 'algorithm', 'cidrlist': 'cidr', 'protocol': 'protocol'}
        self.returns_to_int = {'publicport': 'public_port', 'privateport': 'private_port'}

    def get_rule(self, **kwargs):
        rules = self.query_api('listLoadBalancerRules', **kwargs)
        if rules:
            return rules['loadbalancerrule'][0]

    def _get_common_args(self):
        return {'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id'), 'zoneid': self.get_zone(key='id') if self.module.params.get('zone') else None, 'publicipid': self.get_ip_address(key='id'), 'name': self.module.params.get('name')}

    def present_lb_rule(self):
        required_params = ['algorithm', 'private_port', 'public_port']
        self.module.fail_on_missing_params(required_params=required_params)
        args = self._get_common_args()
        rule = self.get_rule(**args)
        if rule:
            rule = self._update_lb_rule(rule)
        else:
            rule = self._create_lb_rule(rule)
        if rule:
            rule = self.ensure_tags(resource=rule, resource_type='LoadBalancer')
        return rule

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

    def absent_lb_rule(self):
        args = self._get_common_args()
        rule = self.get_rule(**args)
        if rule:
            self.result['changed'] = True
        if rule and (not self.module.check_mode):
            res = self.query_api('deleteLoadBalancerRule', id=rule['id'])
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                self.poll_job(res, 'loadbalancer')
        return rule
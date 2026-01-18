from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
class AnsibleCloudStackLBRuleMember(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackLBRuleMember, self).__init__(module)
        self.returns = {'publicip': 'public_ip', 'algorithm': 'algorithm', 'cidrlist': 'cidr', 'protocol': 'protocol'}
        self.returns_to_int = {'publicport': 'public_port', 'privateport': 'private_port'}

    def get_rule(self):
        args = self._get_common_args()
        args.update({'name': self.module.params.get('name'), 'zoneid': self.get_zone(key='id') if self.module.params.get('zone') else None})
        if self.module.params.get('ip_address'):
            args['publicipid'] = self.get_ip_address(key='id')
        rules = self.query_api('listLoadBalancerRules', **args)
        if rules:
            if len(rules['loadbalancerrule']) > 1:
                self.module.fail_json(msg="More than one rule having name %s. Please pass 'ip_address' as well." % args['name'])
            return rules['loadbalancerrule'][0]
        return None

    def _get_common_args(self):
        return {'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id')}

    def _get_members_of_rule(self, rule):
        res = self.query_api('listLoadBalancerRuleInstances', id=rule['id'])
        return res.get('loadbalancerruleinstance', [])

    def _ensure_members(self, operation):
        if operation not in ['add', 'remove']:
            self.module.fail_json(msg='Bad operation: %s' % operation)
        rule = self.get_rule()
        if not rule:
            self.module.fail_json(msg='Unknown rule: %s' % self.module.params.get('name'))
        existing = {}
        for vm in self._get_members_of_rule(rule=rule):
            existing[vm['name']] = vm['id']
        wanted_names = self.module.params.get('vms')
        if operation == 'add':
            cs_func = 'assignToLoadBalancerRule'
            to_change = set(wanted_names) - set(existing.keys())
        else:
            cs_func = 'removeFromLoadBalancerRule'
            to_change = set(wanted_names) & set(existing.keys())
        if not to_change:
            return rule
        args = self._get_common_args()
        args['fetch_list'] = True
        vms = self.query_api('listVirtualMachines', **args)
        to_change_ids = []
        for name in to_change:
            for vm in vms:
                if vm['name'] == name:
                    to_change_ids.append(vm['id'])
                    break
            else:
                self.module.fail_json(msg='Unknown VM: %s' % name)
        if to_change_ids:
            self.result['changed'] = True
        if to_change_ids and (not self.module.check_mode):
            res = self.query_api(cs_func, id=rule['id'], virtualmachineids=to_change_ids)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                self.poll_job(res)
                rule = self.get_rule()
        return rule

    def add_members(self):
        return self._ensure_members('add')

    def remove_members(self):
        return self._ensure_members('remove')

    def get_result(self, resource):
        super(AnsibleCloudStackLBRuleMember, self).get_result(resource)
        if resource:
            self.result['vms'] = []
            for vm in self._get_members_of_rule(rule=resource):
                self.result['vms'].append(vm['name'])
        return self.result
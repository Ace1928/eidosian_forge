from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
class AnsibleCloudStackNetworkAclRule(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackNetworkAclRule, self).__init__(module)
        self.returns = {'cidrlist': 'cidr', 'action': 'action_policy', 'protocol': 'protocol', 'icmpcode': 'icmp_code', 'icmptype': 'icmp_type', 'number': 'rule_position', 'traffictype': 'traffic_type'}
        self.returns_to_int = {'startport': 'start_port', 'endport': 'end_port'}

    def get_network_acl_rule(self):
        args = {'aclid': self.get_network_acl(key='id'), 'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id')}
        network_acl_rules = self.query_api('listNetworkACLs', **args)
        for acl_rule in network_acl_rules.get('networkacl', []):
            if acl_rule['number'] == self.module.params.get('rule_position'):
                return acl_rule
        return None

    def present_network_acl_rule(self):
        network_acl_rule = self.get_network_acl_rule()
        protocol = self.module.params.get('protocol')
        start_port = self.module.params.get('start_port')
        end_port = self.get_or_fallback('end_port', 'start_port')
        icmp_type = self.module.params.get('icmp_type')
        icmp_code = self.module.params.get('icmp_code')
        if protocol in ['tcp', 'udp'] and (start_port is None or end_port is None):
            self.module.fail_json(msg='protocol is %s but the following are missing: start_port, end_port' % protocol)
        elif protocol == 'icmp' and (icmp_type is None or icmp_code is None):
            self.module.fail_json(msg='protocol is icmp but the following are missing: icmp_type, icmp_code')
        elif protocol == 'by_number' and self.module.params.get('protocol_number') is None:
            self.module.fail_json(msg='protocol is by_number but the following are missing: protocol_number')
        if not network_acl_rule:
            network_acl_rule = self._create_network_acl_rule(network_acl_rule)
        else:
            network_acl_rule = self._update_network_acl_rule(network_acl_rule)
        if network_acl_rule:
            network_acl_rule = self.ensure_tags(resource=network_acl_rule, resource_type='NetworkACL')
        return network_acl_rule

    def absent_network_acl_rule(self):
        network_acl_rule = self.get_network_acl_rule()
        if network_acl_rule:
            self.result['changed'] = True
            args = {'id': network_acl_rule['id']}
            if not self.module.check_mode:
                res = self.query_api('deleteNetworkACL', **args)
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    self.poll_job(res, 'networkacl')
        return network_acl_rule

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

    def _update_network_acl_rule(self, network_acl_rule):
        protocol = self.module.params.get('protocol')
        args = {'id': network_acl_rule['id'], 'action': self.module.params.get('action_policy'), 'protocol': protocol if protocol != 'by_number' else str(self.module.params.get('protocol_number')), 'startport': self.module.params.get('start_port'), 'endport': self.get_or_fallback('end_port', 'start_port'), 'icmpcode': self.module.params.get('icmp_code'), 'icmptype': self.module.params.get('icmp_type'), 'traffictype': self.module.params.get('traffic_type'), 'cidrlist': ','.join(self.module.params.get('cidrs'))}
        if self.has_changed(args, network_acl_rule):
            self.result['changed'] = True
            if not self.module.check_mode:
                res = self.query_api('updateNetworkACLItem', **args)
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    network_acl_rule = self.poll_job(res, 'networkacl')
        return network_acl_rule

    def get_result(self, resource):
        super(AnsibleCloudStackNetworkAclRule, self).get_result(resource)
        if resource:
            if 'cidrlist' in resource:
                self.result['cidrs'] = resource['cidrlist'].split(',') or [resource['cidrlist']]
            if resource['protocol'] not in ['tcp', 'udp', 'icmp', 'all']:
                self.result['protocol_number'] = int(resource['protocol'])
                self.result['protocol'] = 'by_number'
            self.result['action_policy'] = self.result['action_policy'].lower()
            self.result['traffic_type'] = self.result['traffic_type'].lower()
        return self.result
from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.vultr_v2 import AnsibleVultr, vultr_argument_spec
class AnsibleVultrFirewallRule(AnsibleVultr):

    def get_firewall_group(self):
        return self.query_filter_list_by_name(key_name='description', param_key='group', path='/firewalls', result_key='firewall_groups', fail_not_found=True)

    def get_load_balancer(self):
        return self.query_filter_list_by_name(key_name='label', param_key='source', path='/load-balancers', result_key='load_balancers', fail_not_found=True)

    def configure(self):
        self.resource_path = self.resource_path % self.get_firewall_group()['id']
        source = self.module.params.get('source')
        if source is not None and source != 'cloudflare':
            self.module.params['source'] = self.get_load_balancer()['id']
        if self.module.params.get('protocol') not in ('tcp', 'udp') and self.module.params.get('port') is not None:
            self.module.warn('Setting a port (%s) only affects protocols TCP/UDP, but protocol is: %s. Ignoring.' % (self.module.params.get('port'), self.module.params.get('protocol')))
            self.module.params['port'] = None

    def query(self):
        result = dict()
        for resource in self.query_list():
            for key in ('ip_type', 'protocol', 'port', 'source', 'subnet', 'subnet_size'):
                param = self.module.params.get(key)
                if param is None:
                    continue
                if resource.get(key) != param:
                    break
            else:
                result = resource
            if result:
                break
        return result

    def update(self, resource):
        return resource
from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.exoscale import (ExoDns, exo_dns_argument_spec,
class ExoDnsDomain(ExoDns):

    def __init__(self, module):
        super(ExoDnsDomain, self).__init__(module)
        self.name = self.module.params.get('name').lower()

    def get_domain(self):
        domains = self.api_query('/domains', 'GET')
        for z in domains:
            if z['domain']['name'].lower() == self.name:
                return z
        return None

    def present_domain(self):
        domain = self.get_domain()
        data = {'domain': {'name': self.name}}
        if not domain:
            self.result['diff']['after'] = data['domain']
            self.result['changed'] = True
            if not self.module.check_mode:
                domain = self.api_query('/domains', 'POST', data)
        return domain

    def absent_domain(self):
        domain = self.get_domain()
        if domain:
            self.result['diff']['before'] = domain
            self.result['changed'] = True
            if not self.module.check_mode:
                self.api_query('/domains/%s' % domain['domain']['name'], 'DELETE')
        return domain

    def get_result(self, resource):
        if resource:
            self.result['exo_dns_domain'] = resource['domain']
        return self.result
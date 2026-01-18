from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.exoscale import (ExoDns, exo_dns_argument_spec,
def absent_domain(self):
    domain = self.get_domain()
    if domain:
        self.result['diff']['before'] = domain
        self.result['changed'] = True
        if not self.module.check_mode:
            self.api_query('/domains/%s' % domain['domain']['name'], 'DELETE')
    return domain
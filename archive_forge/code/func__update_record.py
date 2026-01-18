from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.exoscale import (ExoDns, exo_dns_argument_spec,
def _update_record(self, record):
    data = {'record': {'name': self.name, 'content': self.content, 'ttl': self.module.params.get('ttl'), 'prio': self.module.params.get('prio')}}
    if self.has_changed(data['record'], record['record']):
        self.result['changed'] = True
        if not self.module.check_mode:
            record = self.api_query('/domains/%s/records/%s' % (self.domain, record['record']['id']), 'PUT', data)
    return record
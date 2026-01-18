from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def _get_common_args(self):
    return {'account': self.get_account(key='name'), 'domainid': self.get_domain(key='id'), 'projectid': self.get_project(key='id'), 'zoneid': self.get_zone(key='id') if self.module.params.get('zone') else None, 'publicipid': self.get_ip_address(key='id'), 'name': self.module.params.get('name')}
from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def _create_zone(self):
    required_params = ['dns1']
    self.module.fail_on_missing_params(required_params=required_params)
    self.result['changed'] = True
    args = self._get_common_zone_args()
    args['domainid'] = self.get_domain(key='id')
    args['securitygroupenabled'] = self.module.params.get('securitygroups_enabled')
    zone = None
    if not self.module.check_mode:
        res = self.query_api('createZone', **args)
        zone = res['zone']
    return zone
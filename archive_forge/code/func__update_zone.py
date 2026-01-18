from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def _update_zone(self):
    zone = self.get_zone()
    args = self._get_common_zone_args()
    args['id'] = zone['id']
    if self.has_changed(args, zone):
        self.result['changed'] = True
        if not self.module.check_mode:
            res = self.query_api('updateZone', **args)
            zone = res['zone']
    return zone
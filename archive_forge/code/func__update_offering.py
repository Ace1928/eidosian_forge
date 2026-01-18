from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def _update_offering(self, service_offering):
    args = {'id': service_offering['id'], 'name': self.module.params.get('name'), 'displaytext': self.get_or_fallback('display_text', 'name')}
    if self.has_changed(args, service_offering):
        self.result['changed'] = True
        if not self.module.check_mode:
            res = self.query_api('updateServiceOffering', **args)
            service_offering = res['serviceoffering']
    return service_offering
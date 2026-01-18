from __future__ import absolute_import, division, print_function
from ansible_collections.netbox.netbox.plugins.module_utils.netbox_utils import (
def _handle_state_new(self, nb_app, nb_endpoint, endpoint_name, data):
    if self.state == 'new':
        self.nb_object, diff = self._create_netbox_object(nb_endpoint, data)
        self.result['msg'] = '%s created' % endpoint_name
        self.result['changed'] = True
        self.result['diff'] = diff
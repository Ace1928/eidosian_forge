from __future__ import absolute_import, division, print_function
from ipaddress import ip_interface
from ansible.module_utils._text import to_text
from ansible_collections.netbox.netbox.plugins.module_utils.netbox_utils import (
def _handle_state_new_present(self, nb_app, nb_endpoint, endpoint_name, name, data):
    if data.get('address'):
        if self.state == 'present':
            self._ensure_object_exists(nb_endpoint, endpoint_name, name, data)
        elif self.state == 'new':
            self.nb_object, diff = self._create_netbox_object(nb_endpoint, data)
            self.result['msg'] = '%s %s created' % (endpoint_name, name)
            self.result['changed'] = True
            self.result['diff'] = diff
    elif self.state == 'present':
        self._ensure_ip_in_prefix_present_on_netif(nb_app, nb_endpoint, data, endpoint_name)
    elif self.state == 'new':
        self._get_new_available_ip_address(nb_app, data, endpoint_name)
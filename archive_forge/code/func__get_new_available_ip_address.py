from __future__ import absolute_import, division, print_function
from ipaddress import ip_interface
from ansible.module_utils._text import to_text
from ansible_collections.netbox.netbox.plugins.module_utils.netbox_utils import (
def _get_new_available_ip_address(self, nb_app, data, endpoint_name):
    prefix_query = self._build_query_params('prefix', data)
    prefix = self._nb_endpoint_get(nb_app.prefixes, prefix_query, data['prefix'])
    if not prefix:
        self.result['changed'] = False
        self.result['msg'] = '%s does not exist - please create first' % data['prefix']
    elif prefix.available_ips.list():
        self.nb_object, diff = self._create_netbox_object(prefix.available_ips, data)
        self.nb_object = self.nb_object.serialize()
        self.result['changed'] = True
        self.result['msg'] = '%s %s created' % (endpoint_name, self.nb_object['address'])
        self.result['diff'] = diff
    else:
        self.result['changed'] = False
        self.result['msg'] = 'No available IPs available within %s' % data['prefix']
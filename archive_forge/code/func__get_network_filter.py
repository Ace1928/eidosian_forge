from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _get_network_filter(self):
    network_filter = None
    if self.param('network_filter') == '' or self.param('pass_through') == 'enabled':
        network_filter = otypes.NetworkFilter()
    elif self.param('network_filter'):
        network_filter = otypes.NetworkFilter(id=self._get_network_filter_id())
    return network_filter
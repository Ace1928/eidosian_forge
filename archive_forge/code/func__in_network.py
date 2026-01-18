from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddress_utils import (
from ansible_collections.ansible.utils.plugins.plugin_utils.base.utils import _validate_args
@_need_ipaddress
def _in_network(ip, network):
    """Test if an address or network is in a network"""
    params = {'ip': ip, 'network': network}
    _validate_args('in_network', DOCUMENTATION, params)
    try:
        return _is_subnet_of(ip_network(ip), ip_network(network))
    except Exception:
        return False
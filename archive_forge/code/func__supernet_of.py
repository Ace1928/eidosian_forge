from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddress_utils import (
from ansible_collections.ansible.utils.plugins.plugin_utils.base.utils import _validate_args
@_need_ipaddress
def _supernet_of(network_a, network_b):
    """Test if an network is a supernet of another network"""
    params = {'network_a': network_a, 'network_b': network_b}
    _validate_args('supernet_of', DOCUMENTATION, params)
    try:
        return _is_subnet_of(ip_network(network_b), ip_network(network_a))
    except Exception:
        return False
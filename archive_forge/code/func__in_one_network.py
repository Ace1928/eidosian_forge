from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddress_utils import (
from ansible_collections.ansible.utils.plugins.test.in_network import _in_network
def _in_one_network(ip, networks):
    """Test if an IP or network is in one network"""
    params = {'ip': ip, 'networks': networks}
    _validate_args('in_one_network', DOCUMENTATION, params)
    bools = [_in_network(ip, network) for network in networks]
    if bools.count(True) == 1:
        return True
    return False
from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.utils.plugins.plugin_utils.base.utils import _validate_args
from ansible_collections.ansible.utils.plugins.test.in_network import _in_network
def _in_any_network(ip, networks):
    """Test if an IP or network is in any network"""
    params = {'ip': ip, 'networks': networks}
    _validate_args('in_any_network', DOCUMENTATION, params)
    bools = [_in_network(ip, network) for network in networks]
    if True in bools:
        return True
    return False
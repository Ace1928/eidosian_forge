from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddress_utils import (
@_need_ipaddress
def _ipv4(ip):
    """Test if something in an IPv4 address or network"""
    params = {'ip': ip}
    _validate_args('ipv4', DOCUMENTATION, params)
    try:
        return ip_network(ip).version == 4
    except Exception:
        return False
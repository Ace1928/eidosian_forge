from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddress_utils import (
@_need_ipaddress
def _ipv6_ipv4_mapped(ip):
    """Test if something appears to be a mapped IPv6 to IPv4 mapped address"""
    params = {'ip': ip}
    _validate_args('ipv6_ipv4_mapped', DOCUMENTATION, params)
    try:
        if ip_address(ip).ipv4_mapped is None:
            return False
        return True
    except Exception:
        return False
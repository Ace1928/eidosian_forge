from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddress_utils import (
from ansible_collections.ansible.utils.plugins.plugin_utils.base.utils import _validate_args
@_need_ipaddress
def _multicast(ip):
    """Test for a multicast IP address"""
    params = {'ip': ip}
    _validate_args('multicast', DOCUMENTATION, params)
    try:
        return ip_address(ip).is_multicast
    except Exception:
        return False
from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.utils.plugins.plugin_utils.base.ipaddress_utils import (
from ansible_collections.ansible.utils.plugins.plugin_utils.base.utils import _validate_args
@_need_ipaddress
def _private(ip):
    """Test if an IP address is private"""
    params = {'ip': ip}
    _validate_args('private', DOCUMENTATION, params)
    try:
        return ip_address(ip).is_private
    except Exception:
        return False
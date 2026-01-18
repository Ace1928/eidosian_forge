from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
def _get_ipv4orv6_address(ip_address, module):
    """
    return IPV4Adress or IPV6Address object
    """
    _check_ipaddress_is_present(module)
    ip_addr = u'%s' % ip_address
    try:
        return ipaddress.ip_address(ip_addr)
    except ValueError as exc:
        error = 'Error: Invalid IP address value %s - %s' % (ip_address, to_native(exc))
        module.fail_json(msg=error)
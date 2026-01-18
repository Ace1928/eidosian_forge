from __future__ import absolute_import, division, print_function
import types
from ansible.errors import AnsibleFilterError
from ansible.module_utils.basic import missing_required_lib
from ansible.utils.display import Display
def _public_query(v, value):
    v_ip = netaddr.IPAddress(str(v.ip))
    if all([v_ip.is_unicast(), not v_ip.is_private(), not v_ip.is_loopback(), not v_ip.is_netmask(), not v_ip.is_hostmask()]):
        return value
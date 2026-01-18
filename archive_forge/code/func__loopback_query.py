from __future__ import absolute_import, division, print_function
import types
from ansible.errors import AnsibleFilterError
from ansible.module_utils.basic import missing_required_lib
from ansible.utils.display import Display
def _loopback_query(v, value):
    v_ip = netaddr.IPAddress(str(v.ip))
    if v_ip.is_loopback():
        return value
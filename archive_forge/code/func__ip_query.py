from __future__ import absolute_import, division, print_function
import types
from ansible.errors import AnsibleFilterError
from ansible.module_utils.basic import missing_required_lib
from ansible.utils.display import Display
def _ip_query(v):
    if v.size == 1:
        return str(v.ip)
    if v.size > 1:
        if v.ip != v.network or not v.broadcast:
            return str(v.ip)
        elif v.version == 6 and v.ip == v.network:
            return str(v.ip)
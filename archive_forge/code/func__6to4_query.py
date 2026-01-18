from __future__ import absolute_import, division, print_function
import types
from ansible.errors import AnsibleFilterError
from ansible.module_utils.basic import missing_required_lib
from ansible.utils.display import Display
def _6to4_query(v, vtype, value):
    if v.version == 4:
        if v.size == 1:
            ipconv = str(v.ip)
        elif v.size > 1:
            if v.ip != v.network:
                ipconv = str(v.ip)
            else:
                return False
        if ipaddr(ipconv, 'public') or ipaddr(ipconv, 'private'):
            numbers = list(map(int, ipconv.split('.')))
        try:
            return '2002:{:02x}{:02x}:{:02x}{:02x}::1/48'.format(*numbers)
        except Exception:
            pass
    elif v.version == 6:
        if vtype == 'address':
            if ipaddr(str(v), '2002::/16'):
                return value
        elif vtype == 'network':
            if v.ip != v.network:
                if ipaddr(str(v.ip), '2002::/16'):
                    return value
    return False
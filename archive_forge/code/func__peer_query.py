from __future__ import absolute_import, division, print_function
import types
from ansible.errors import AnsibleFilterError
from ansible.module_utils.basic import missing_required_lib
from ansible.utils.display import Display
def _peer_query(v, vtype):
    if vtype == 'address':
        raise AnsibleFilterError('Not a network address')
    elif vtype == 'network':
        if v.size == 2:
            return str(netaddr.IPAddress(int(v.ip) ^ 1))
        if v.size == 4:
            if int(v.ip) % 4 == 0:
                raise AnsibleFilterError('Network address of /30 has no peer')
            if int(v.ip) % 4 == 3:
                raise AnsibleFilterError('Broadcast address of /30 has no peer')
            return str(netaddr.IPAddress(int(v.ip) ^ 3))
        raise AnsibleFilterError('Not a point-to-point network')
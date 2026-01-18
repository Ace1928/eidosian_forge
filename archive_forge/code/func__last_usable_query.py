from __future__ import absolute_import, division, print_function
import types
from ansible.errors import AnsibleFilterError
from ansible.module_utils.basic import missing_required_lib
from ansible.utils.display import Display
def _last_usable_query(v, vtype):
    if vtype == 'address':
        raise AnsibleFilterError('Not a network address')
    elif vtype == 'network':
        if v.size > 1:
            first_usable, last_usable = _first_last(v)
            return str(netaddr.IPAddress(last_usable))
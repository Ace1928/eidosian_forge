from __future__ import absolute_import, division, print_function
import types
from ansible.errors import AnsibleFilterError
from ansible.module_utils.basic import missing_required_lib
from ansible.utils.display import Display
def _empty_ipaddr_query(v, vtype):
    if v:
        if vtype == 'address':
            return str(v.ip)
        elif vtype == 'network':
            return str(v)
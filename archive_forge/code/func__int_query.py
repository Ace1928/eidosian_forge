from __future__ import absolute_import, division, print_function
import types
from ansible.errors import AnsibleFilterError
from ansible.module_utils.basic import missing_required_lib
from ansible.utils.display import Display
def _int_query(v, vtype):
    if vtype == 'address':
        return int(v.ip)
    elif vtype == 'network':
        return str(int(v.ip)) + '/' + str(int(v.prefixlen))
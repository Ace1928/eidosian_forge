from __future__ import absolute_import, division, print_function
import types
from ansible.errors import AnsibleFilterError
from ansible.module_utils.basic import missing_required_lib
from ansible.utils.display import Display
def _ipv6_query(v, value):
    if v.version == 4:
        return str(v.ipv6())
    else:
        return value
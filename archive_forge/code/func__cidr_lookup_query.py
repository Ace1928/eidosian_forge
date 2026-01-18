from __future__ import absolute_import, division, print_function
import types
from ansible.errors import AnsibleFilterError
from ansible.module_utils.basic import missing_required_lib
from ansible.utils.display import Display
def _cidr_lookup_query(v, iplist, value):
    try:
        if v in iplist:
            return value
    except Exception:
        return False
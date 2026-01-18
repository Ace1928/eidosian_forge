from __future__ import absolute_import, division, print_function
import types
from ansible.errors import AnsibleFilterError
from ansible.module_utils.basic import missing_required_lib
from ansible.utils.display import Display
def _ipv4_query(v, value):
    if v.version == 6:
        try:
            return str(v.ipv4())
        except Exception:
            return False
    else:
        return value
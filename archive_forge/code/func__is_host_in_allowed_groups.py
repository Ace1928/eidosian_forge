from __future__ import (absolute_import, division, print_function)
import re
from ansible.errors import AnsibleError
from ansible.plugins.lookup import LookupBase
from ansible.utils.display import Display
def _is_host_in_allowed_groups(self, host_groups):
    if 'all' in self._groups:
        return True
    group_intersection = [host_group_name for host_group_name in host_groups if host_group_name in self._groups]
    if group_intersection:
        return True
    return False
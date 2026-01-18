from __future__ import (absolute_import, division, print_function)
import socket
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_text
from ansible.plugins.inventory import BaseInventoryPlugin, Cacheable, to_safe_group_name
from ansible.module_utils.six import text_type
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
def _exclude_profile(self, profile):
    if self.include_profiles:
        return profile not in self.include_profiles
    else:
        return profile in self.exclude_profiles
from __future__ import (absolute_import, division, print_function)
import hashlib
import os
import string
from collections.abc import Mapping
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.inventory.group import to_safe_group_name as original_safe
from ansible.parsing.utils.addresses import parse_address
from ansible.plugins import AnsiblePlugin
from ansible.plugins.cache import CachePluginAdjudicator as CacheObject
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import string_types
from ansible.template import Templar
from ansible.utils.display import Display
from ansible.utils.vars import combine_vars, load_extra_vars
def _populate_host_vars(self, hosts, variables, group=None, port=None):
    if not isinstance(variables, Mapping):
        raise AnsibleParserError('Invalid data from file, expected dictionary and got:\n\n%s' % to_native(variables))
    for host in hosts:
        self.inventory.add_host(host, group=group, port=port)
        for k in variables:
            self.inventory.set_variable(host, k, variables[k])
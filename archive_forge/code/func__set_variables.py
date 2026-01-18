from __future__ import (absolute_import, division, print_function)
import os
from subprocess import Popen, PIPE
from ansible.errors import AnsibleParserError
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.common._collections_compat import MutableMapping
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
from ansible.module_utils.common.process import get_bin_path
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
def _set_variables(self, hostvars):
    for host in hostvars:
        query = self.get_option('query')
        if query and isinstance(query, MutableMapping):
            for varname in query:
                hostvars[host][varname] = self._query_vbox_data(host, query[varname])
        strict = self.get_option('strict')
        self._set_composite_vars(self.get_option('compose'), hostvars[host], host, strict=strict)
        for key in hostvars[host]:
            self.inventory.set_variable(host, key, hostvars[host][key])
        self._add_host_to_composed_groups(self.get_option('groups'), hostvars[host], host, strict=strict)
        self._add_host_to_keyed_groups(self.get_option('keyed_groups'), hostvars[host], host, strict=strict)
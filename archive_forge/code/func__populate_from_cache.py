from __future__ import (absolute_import, division, print_function)
import os
from subprocess import Popen, PIPE
from ansible.errors import AnsibleParserError
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.common._collections_compat import MutableMapping
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
from ansible.module_utils.common.process import get_bin_path
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
def _populate_from_cache(self, source_data):
    source_data = make_unsafe(source_data)
    hostvars = source_data.pop('_meta', {}).get('hostvars', {})
    for group in source_data:
        if group == 'all':
            continue
        else:
            group = self.inventory.add_group(group)
            hosts = source_data[group].get('hosts', [])
            for host in hosts:
                self._populate_host_vars([host], hostvars.get(host, {}), group)
            self.inventory.add_child('all', group)
    if not source_data:
        for host in hostvars:
            self.inventory.add_host(host)
            self._populate_host_vars([host], hostvars.get(host, {}))
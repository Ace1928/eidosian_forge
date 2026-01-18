from __future__ import (absolute_import, division, print_function)
import os
from subprocess import Popen, PIPE
from ansible.errors import AnsibleParserError
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.common._collections_compat import MutableMapping
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
from ansible.module_utils.common.process import get_bin_path
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
def _ungrouped_host(self, host, inventory):

    def find_host(host, inventory):
        for k, v in inventory.items():
            if k == '_meta':
                continue
            if isinstance(v, dict):
                yield self._ungrouped_host(host, v)
            elif isinstance(v, list):
                yield (host not in v)
        yield True
    return all(find_host(host, inventory))
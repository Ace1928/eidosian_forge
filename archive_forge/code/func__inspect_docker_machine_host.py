from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.common.process import get_bin_path
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
from ansible.utils.display import Display
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
from ansible_collections.community.library_inventory_filtering_v1.plugins.plugin_utils.inventory_filter import parse_filters, filter_host
import json
import re
import subprocess
def _inspect_docker_machine_host(self, node):
    try:
        inspect_lines = self._run_command(['inspect', node])
    except subprocess.CalledProcessError:
        return None
    return json.loads(inspect_lines)
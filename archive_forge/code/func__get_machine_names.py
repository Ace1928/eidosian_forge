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
def _get_machine_names(self):
    ls_command = ['ls', '-q']
    if self.get_option('running_required'):
        ls_command.extend(['--filter', 'state=Running'])
    try:
        ls_lines = self._run_command(ls_command)
    except subprocess.CalledProcessError:
        return []
    return ls_lines.splitlines()
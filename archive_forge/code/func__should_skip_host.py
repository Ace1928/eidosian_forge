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
def _should_skip_host(self, machine_name, env_var_tuples, daemon_env):
    if not env_var_tuples:
        warning_prefix = 'Unable to fetch Docker daemon env vars from Docker Machine for host {0}'.format(machine_name)
        if daemon_env in ('require', 'require-silently'):
            if daemon_env == 'require':
                display.warning('{0}: host will be skipped'.format(warning_prefix))
            return True
        elif daemon_env == 'optional':
            display.warning('{0}: host will lack dm_DOCKER_xxx variables'.format(warning_prefix))
    return False
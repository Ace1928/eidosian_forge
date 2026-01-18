from __future__ import (absolute_import, division, print_function)
from ansible.cli import CLI
import sys
import argparse
from ansible import constants as C
from ansible import context
from ansible.cli.arguments import option_helpers as opt_help
from ansible.errors import AnsibleError, AnsibleOptionsError
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.utils.vars import combine_vars
from ansible.utils.display import Display
from ansible.vars.plugins import get_vars_from_inventory_sources, get_vars_from_path
def _graph_group(self, group, depth=0):
    result = [self._graph_name('@%s:' % group.name, depth)]
    depth = depth + 1
    for kid in group.child_groups:
        result.extend(self._graph_group(kid, depth))
    if group.name != 'all':
        for host in group.hosts:
            result.append(self._graph_name(host.name, depth))
            if context.CLIARGS['show_vars']:
                result.extend(self._show_vars(self._get_host_variables(host), depth + 1))
    if context.CLIARGS['show_vars']:
        result.extend(self._show_vars(self._get_group_variables(group), depth))
    return result
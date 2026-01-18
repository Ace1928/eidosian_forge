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
@staticmethod
def _show_vars(dump, depth):
    result = []
    for name, val in sorted(dump.items()):
        result.append(InventoryCLI._graph_name('{%s = %s}' % (name, val), depth))
    return result
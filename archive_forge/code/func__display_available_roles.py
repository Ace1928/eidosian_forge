from __future__ import (absolute_import, division, print_function)
from ansible.cli import CLI
import pkgutil
import os
import os.path
import re
import textwrap
import traceback
import ansible.plugins.loader as plugin_loader
from pathlib import Path
from ansible import constants as C
from ansible import context
from ansible.cli.arguments import option_helpers as opt_help
from ansible.collections.list import list_collection_dirs
from ansible.errors import AnsibleError, AnsibleOptionsError, AnsibleParserError, AnsiblePluginNotFound
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.common.json import json_dump
from ansible.module_utils.common.yaml import yaml_dump
from ansible.module_utils.compat import importlib
from ansible.module_utils.six import string_types
from ansible.parsing.plugin_docs import read_docstub
from ansible.parsing.utils.yaml import from_yaml
from ansible.parsing.yaml.dumper import AnsibleDumper
from ansible.plugins.list import list_plugins
from ansible.plugins.loader import action_loader, fragment_loader
from ansible.utils.collection_loader import AnsibleCollectionConfig, AnsibleCollectionRef
from ansible.utils.collection_loader._collection_finder import _get_collection_name_from_path
from ansible.utils.display import Display
from ansible.utils.plugin_docs import get_plugin_docs, get_docstring, get_versioned_doclink
def _display_available_roles(self, list_json):
    """Display all roles we can find with a valid argument specification.

        Output is: fqcn role name, entry point, short description
        """
    roles = list(list_json.keys())
    entry_point_names = set()
    for role in roles:
        for entry_point in list_json[role]['entry_points'].keys():
            entry_point_names.add(entry_point)
    max_role_len = 0
    max_ep_len = 0
    if roles:
        max_role_len = max((len(x) for x in roles))
    if entry_point_names:
        max_ep_len = max((len(x) for x in entry_point_names))
    linelimit = display.columns - max_role_len - max_ep_len - 5
    text = []
    for role in sorted(roles):
        for entry_point, desc in list_json[role]['entry_points'].items():
            if len(desc) > linelimit:
                desc = desc[:linelimit] + '...'
            text.append('%-*s %-*s %s' % (max_role_len, role, max_ep_len, entry_point, desc))
    DocCLI.pager('\n'.join(text))
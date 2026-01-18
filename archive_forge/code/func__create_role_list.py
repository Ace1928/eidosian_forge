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
def _create_role_list(self, fail_on_errors=True):
    """Return a dict describing the listing of all roles with arg specs.

        :param role_paths: A tuple of one or more role paths.

        :returns: A dict indexed by role name, with 'collection' and 'entry_points' keys per role.

        Example return:

            results = {
               'roleA': {
                  'collection': '',
                  'entry_points': {
                     'main': 'Short description for main'
                  }
               },
               'a.b.c.roleB': {
                  'collection': 'a.b.c',
                  'entry_points': {
                     'main': 'Short description for main',
                     'alternate': 'Short description for alternate entry point'
                  }
               'x.y.z.roleB': {
                  'collection': 'x.y.z',
                  'entry_points': {
                     'main': 'Short description for main',
                  }
               },
            }
        """
    roles_path = self._get_roles_path()
    collection_filter = self._get_collection_filter()
    if not collection_filter:
        roles = self._find_all_normal_roles(roles_path)
    else:
        roles = []
    collroles = self._find_all_collection_roles(collection_filter=collection_filter)
    result = {}
    for role, role_path in roles:
        try:
            argspec = self._load_argspec(role, role_path=role_path)
            fqcn, summary = self._build_summary(role, '', argspec)
            result[fqcn] = summary
        except Exception as e:
            if fail_on_errors:
                raise
            result[role] = {'error': 'Error while loading role argument spec: %s' % to_native(e)}
    for role, collection, collection_path in collroles:
        try:
            argspec = self._load_argspec(role, collection_path=collection_path)
            fqcn, summary = self._build_summary(role, collection, argspec)
            result[fqcn] = summary
        except Exception as e:
            if fail_on_errors:
                raise
            result['%s.%s' % (collection, role)] = {'error': 'Error while loading role argument spec: %s' % to_native(e)}
    return result
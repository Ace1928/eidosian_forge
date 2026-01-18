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
def _create_role_doc(self, role_names, entry_point=None, fail_on_errors=True):
    """
        :param role_names: A tuple of one or more role names.
        :param role_paths: A tuple of one or more role paths.
        :param entry_point: A role entry point name for filtering.
        :param fail_on_errors: When set to False, include errors in the JSON output instead of raising errors

        :returns: A dict indexed by role name, with 'collection', 'entry_points', and 'path' keys per role.
        """
    roles_path = self._get_roles_path()
    roles = self._find_all_normal_roles(roles_path, name_filters=role_names)
    collroles = self._find_all_collection_roles(name_filters=role_names)
    result = {}
    for role, role_path in roles:
        try:
            argspec = self._load_argspec(role, role_path=role_path)
            fqcn, doc = self._build_doc(role, role_path, '', argspec, entry_point)
            if doc:
                result[fqcn] = doc
        except Exception as e:
            result[role] = {'error': 'Error while processing role: %s' % to_native(e)}
    for role, collection, collection_path in collroles:
        try:
            argspec = self._load_argspec(role, collection_path=collection_path)
            fqcn, doc = self._build_doc(role, collection_path, collection, argspec, entry_point)
            if doc:
                result[fqcn] = doc
        except Exception as e:
            result['%s.%s' % (collection, role)] = {'error': 'Error while processing role: %s' % to_native(e)}
    return result
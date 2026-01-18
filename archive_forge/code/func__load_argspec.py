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
def _load_argspec(self, role_name, collection_path=None, role_path=None):
    """Load the role argument spec data from the source file.

        :param str role_name: The name of the role for which we want the argspec data.
        :param str collection_path: Path to the collection containing the role. This
            will be None for standard roles.
        :param str role_path: Path to the standard role. This will be None for
            collection roles.

        We support two files containing the role arg spec data: either meta/main.yml
        or meta/argument_spec.yml. The argument_spec.yml file will take precedence
        over the meta/main.yml file, if it exists. Data is NOT combined between the
        two files.

        :returns: A dict of all data underneath the ``argument_specs`` top-level YAML
            key in the argspec data file. Empty dict is returned if there is no data.
        """
    if collection_path:
        meta_path = os.path.join(collection_path, 'roles', role_name, 'meta')
    elif role_path:
        meta_path = os.path.join(role_path, 'meta')
    else:
        raise AnsibleError("A path is required to load argument specs for role '%s'" % role_name)
    path = None
    for specfile in self.ROLE_ARGSPEC_FILES:
        full_path = os.path.join(meta_path, specfile)
        if os.path.exists(full_path):
            path = full_path
            break
    if path is None:
        return {}
    try:
        with open(path, 'r') as f:
            data = from_yaml(f.read(), file_name=path)
            if data is None:
                data = {}
            return data.get('argument_specs', {})
    except (IOError, OSError) as e:
        raise AnsibleParserError("An error occurred while trying to read the file '%s': %s" % (path, to_native(e)), orig_exc=e)
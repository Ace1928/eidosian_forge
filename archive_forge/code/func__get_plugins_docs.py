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
def _get_plugins_docs(self, plugin_type, names, fail_ok=False, fail_on_errors=True):
    loader = DocCLI._prep_loader(plugin_type)
    plugin_docs = {}
    for plugin in names:
        doc = {}
        try:
            doc, plainexamples, returndocs, metadata = get_plugin_docs(plugin, plugin_type, loader, fragment_loader, context.CLIARGS['verbosity'] > 0)
        except AnsiblePluginNotFound as e:
            display.warning(to_native(e))
            continue
        except Exception as e:
            if not fail_on_errors:
                plugin_docs[plugin] = {'error': 'Missing documentation or could not parse documentation: %s' % to_native(e)}
                continue
            display.vvv(traceback.format_exc())
            msg = '%s %s missing documentation (or could not parse documentation): %s\n' % (plugin_type, plugin, to_native(e))
            if fail_ok:
                display.warning(msg)
            else:
                raise AnsibleError(msg)
        if not doc:
            if not fail_on_errors:
                plugin_docs[plugin] = {'error': 'No valid documentation found'}
            continue
        docs = DocCLI._combine_plugin_doc(plugin, plugin_type, doc, plainexamples, returndocs, metadata)
        if not fail_on_errors:
            try:
                json_dump(docs)
            except Exception as e:
                plugin_docs[plugin] = {'error': 'Cannot serialize documentation as JSON: %s' % to_native(e)}
                continue
        plugin_docs[plugin] = docs
    return plugin_docs
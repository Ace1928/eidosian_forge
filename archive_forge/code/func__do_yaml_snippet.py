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
def _do_yaml_snippet(doc):
    text = []
    mdesc = DocCLI.tty_ify(doc['short_description'])
    module = doc.get('module')
    if module:
        text.append('- name: %s' % mdesc)
        text.append('  %s:' % module)
    else:
        text.append('# %s:' % doc.get('plugin', doc.get('name')))
    pad = 29
    subdent = '# '.rjust(pad + 2)
    limit = display.columns - pad
    for o in sorted(doc['options'].keys()):
        opt = doc['options'][o]
        if isinstance(opt['description'], string_types):
            desc = DocCLI.tty_ify(opt['description'])
        else:
            desc = DocCLI.tty_ify(' '.join(opt['description']))
        required = opt.get('required', False)
        if not isinstance(required, bool):
            raise ValueError("Incorrect value for 'Required', a boolean is needed: %s" % required)
        o = '%s:' % o
        if module:
            if required:
                desc = '(required) %s' % desc
            text.append('      %-20s   # %s' % (o, textwrap.fill(desc, limit, subsequent_indent=subdent)))
        else:
            if required:
                default = '(required)'
            else:
                default = opt.get('default', 'None')
            text.append('%s %-9s # %s' % (o, default, textwrap.fill(desc, limit, subsequent_indent=subdent, max_lines=3)))
    return text
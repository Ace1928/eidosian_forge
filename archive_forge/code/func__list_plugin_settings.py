from __future__ import (absolute_import, division, print_function)
from ansible.cli import CLI
import os
import yaml
import shlex
import subprocess
from collections.abc import Mapping
from ansible import context
import ansible.plugins.loader as plugin_loader
from ansible import constants as C
from ansible.cli.arguments import option_helpers as opt_help
from ansible.config.manager import ConfigManager, Setting
from ansible.errors import AnsibleError, AnsibleOptionsError
from ansible.module_utils.common.text.converters import to_native, to_text, to_bytes
from ansible.module_utils.common.json import json_dump
from ansible.module_utils.six import string_types
from ansible.parsing.quoting import is_quoted
from ansible.parsing.yaml.dumper import AnsibleDumper
from ansible.utils.color import stringc
from ansible.utils.display import Display
from ansible.utils.path import unfrackpath
def _list_plugin_settings(self, ptype, plugins=None):
    entries = {}
    loader = getattr(plugin_loader, '%s_loader' % ptype)
    if plugins:
        plugin_cs = []
        for plugin in plugins:
            p = loader.get(plugin, class_only=True)
            if p is None:
                display.warning('Skipping %s as we could not find matching plugin' % plugin)
            else:
                plugin_cs.append(p)
    else:
        plugin_cs = loader.all(class_only=True)
    for plugin in plugin_cs:
        finalname = name = plugin._load_name
        if name.startswith('_'):
            if os.path.islink(plugin._original_path):
                continue
            else:
                finalname = name.replace('_', '', 1) + ' (DEPRECATED)'
        entries[finalname] = self.config.get_configuration_definitions(ptype, name)
    return entries
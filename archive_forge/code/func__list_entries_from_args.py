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
def _list_entries_from_args(self):
    """
        build a dict with the list requested configs
        """
    config_entries = {}
    if context.CLIARGS['type'] in ('base', 'all'):
        config_entries = self.config.get_configuration_definitions(ignore_private=True)
    if context.CLIARGS['type'] != 'base':
        config_entries['PLUGINS'] = {}
    if context.CLIARGS['type'] == 'all':
        for ptype in C.CONFIGURABLE_PLUGINS:
            config_entries['PLUGINS'][ptype.upper()] = self._list_plugin_settings(ptype)
    elif context.CLIARGS['type'] != 'base':
        config_entries['PLUGINS'][context.CLIARGS['type']] = self._list_plugin_settings(context.CLIARGS['type'], context.CLIARGS['args'])
    return config_entries
from __future__ import (absolute_import, division, print_function)
import atexit
import configparser
import os
import os.path
import sys
import stat
import tempfile
from collections import namedtuple
from collections.abc import Mapping, Sequence
from jinja2.nativetypes import NativeEnvironment
from ansible.errors import AnsibleOptionsError, AnsibleError
from ansible.module_utils.common.text.converters import to_text, to_bytes, to_native
from ansible.module_utils.common.yaml import yaml_load
from ansible.module_utils.six import string_types
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.parsing.quoting import unquote
from ansible.parsing.yaml.objects import AnsibleVaultEncryptedUnicode
from ansible.utils import py3compat
from ansible.utils.path import cleanup_tmp_file, makedirs_safe, unfrackpath
def _read_config_yaml_file(self, yml_file):
    yml_file = to_bytes(yml_file)
    if os.path.exists(yml_file):
        with open(yml_file, 'rb') as config_def:
            return yaml_load(config_def) or {}
    raise AnsibleError('Missing base YAML definition file (bad install?): %s' % to_native(yml_file))
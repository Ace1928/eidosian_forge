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
def _loop_entries(self, container, entry_list):
    """ repeat code for value entry assignment """
    value = None
    origin = None
    for entry in entry_list:
        name = entry.get('name')
        try:
            temp_value = container.get(name, None)
        except UnicodeEncodeError:
            self.WARNINGS.add(u'value for config entry {0} contains invalid characters, ignoring...'.format(to_text(name)))
            continue
        if temp_value is not None:
            if isinstance(temp_value, AnsibleVaultEncryptedUnicode):
                temp_value = to_text(temp_value, errors='surrogate_or_strict')
            value = temp_value
            origin = name
            if 'deprecated' in entry:
                self.DEPRECATED.append((entry['name'], entry['deprecated']))
    return (value, origin)
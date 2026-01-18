from __future__ import absolute_import, division, print_function
import abc
import os
import re
import shlex
from functools import partial
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.text.formatters import human_to_bytes
from ansible.module_utils.six import string_types
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._platform import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import (
def _preprocess_tmpfs(module, values):
    if 'tmpfs' not in values:
        return values
    result = {}
    for tmpfs_spec in values['tmpfs']:
        split_spec = tmpfs_spec.split(':', 1)
        if len(split_spec) > 1:
            result[split_spec[0]] = split_spec[1]
        else:
            result[split_spec[0]] = ''
    return {'tmpfs': result}
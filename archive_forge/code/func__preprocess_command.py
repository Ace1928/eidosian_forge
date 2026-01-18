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
def _preprocess_command(module, values):
    if 'command' not in values:
        return values
    value = values['command']
    if module.params['command_handling'] == 'correct':
        if value is not None:
            if not isinstance(value, list):
                value = shlex.split(to_text(value, errors='surrogate_or_strict'))
            value = [to_text(x, errors='surrogate_or_strict') for x in value]
    elif value:
        if isinstance(value, list):
            value = shlex.split(' '.join([to_text(x, errors='surrogate_or_strict') for x in value]))
            value = [to_text(x, errors='surrogate_or_strict') for x in value]
        else:
            value = shlex.split(to_text(value, errors='surrogate_or_strict'))
            value = [to_text(x, errors='surrogate_or_strict') for x in value]
    else:
        return {}
    return {'command': value}
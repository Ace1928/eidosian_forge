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
def _is_volume_permissions(mode):
    for part in mode.split(','):
        if part not in ('rw', 'ro', 'z', 'Z', 'consistent', 'delegated', 'cached', 'rprivate', 'private', 'rshared', 'shared', 'rslave', 'slave', 'nocopy'):
            return False
    return True
from __future__ import (absolute_import, division, print_function)
import os
import os.path
import random
import re
import shlex
import time
from collections.abc import Mapping, Sequence
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import text_type, string_types
from ansible.plugins import AnsiblePlugin
def _normalize_system_tmpdirs(self):
    normalized_paths = [d.rstrip('/') for d in self.get_option('system_tmpdirs')]
    if not all((os.path.isabs(d) for d in normalized_paths)):
        raise AnsibleError('The configured system_tmpdirs contains a relative path: {0}. All system_tmpdirs must be absolute'.format(to_native(normalized_paths)))
    self.set_option('system_tmpdirs', normalized_paths)
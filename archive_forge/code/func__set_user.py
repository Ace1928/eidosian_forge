from __future__ import (absolute_import, division, print_function)
import os
import shlex
import shutil
import subprocess
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_bytes, to_native, to_text
from ansible.plugins.connection import ConnectionBase, ensure_connect
from ansible.utils.display import Display
def _set_user(self):
    self._buildah(b'config', [b'--user=' + to_bytes(self.user, errors='surrogate_or_strict')])
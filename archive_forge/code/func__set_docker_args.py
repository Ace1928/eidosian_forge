from __future__ import (absolute_import, division, print_function)
import fcntl
import os
import os.path
import subprocess
import re
from ansible.compat import selectors
from ansible.errors import AnsibleError, AnsibleFileNotFound
from ansible.module_utils.six.moves import shlex_quote
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.plugins.connection import ConnectionBase, BUFSIZE
from ansible.utils.display import Display
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
def _set_docker_args(self):
    del self._docker_args[:]
    extra_args = self.get_option('docker_extra_args') or getattr(self._play_context, 'docker_extra_args', '')
    if extra_args:
        self._docker_args += extra_args.split(' ')
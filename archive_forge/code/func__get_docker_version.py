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
def _get_docker_version(self):
    cmd, cmd_output, err, returncode = self._old_docker_version()
    if returncode == 0:
        for line in to_text(cmd_output, errors='surrogate_or_strict').split(u'\n'):
            if line.startswith(u'Server version:'):
                return self._sanitize_version(line.split()[2])
    cmd, cmd_output, err, returncode = self._new_docker_version()
    if returncode:
        raise AnsibleError('Docker version check (%s) failed: %s' % (to_native(cmd), to_native(err)))
    return self._sanitize_version(to_text(cmd_output, errors='surrogate_or_strict'))
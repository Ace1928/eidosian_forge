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
def _new_docker_version(self):
    cmd_args = self._docker_args
    new_version_subcommand = ['version', '--format', "'{{.Server.Version}}'"]
    new_docker_cmd = [self.docker_cmd] + cmd_args + new_version_subcommand
    p = subprocess.Popen(new_docker_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cmd_output, err = p.communicate()
    return (new_docker_cmd, to_native(cmd_output), err, p.returncode)
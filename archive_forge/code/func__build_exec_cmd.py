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
def _build_exec_cmd(self, cmd):
    """ Build the local docker exec command to run cmd on remote_host

            If remote_user is available and is supported by the docker
            version we are using, it will be provided to docker exec.
        """
    local_cmd = [self.docker_cmd]
    if self._docker_args:
        local_cmd += self._docker_args
    local_cmd += [b'exec']
    if self.remote_user is not None:
        local_cmd += [b'-u', self.remote_user]
    local_cmd += [b'-i', self.get_option('remote_addr')] + cmd
    return local_cmd
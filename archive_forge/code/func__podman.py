from __future__ import (absolute_import, division, print_function)
import os
import shlex
import shutil
import subprocess
from ansible.module_utils.common.process import get_bin_path
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_bytes, to_native
from ansible.plugins.connection import ConnectionBase, ensure_connect
from ansible.utils.display import Display
def _podman(self, cmd, cmd_args=None, in_data=None, use_container_id=True):
    """
        run podman executable

        :param cmd: podman's command to execute (str or list)
        :param cmd_args: list of arguments to pass to the command (list of str/bytes)
        :param in_data: data passed to podman's stdin
        :param use_container_id: whether to append the container ID to the command
        :return: return code, stdout, stderr
        """
    podman_exec = self.get_option('podman_executable')
    try:
        podman_cmd = get_bin_path(podman_exec)
    except ValueError:
        raise AnsibleError('%s command not found in PATH' % podman_exec)
    if not podman_cmd:
        raise AnsibleError('%s command not found in PATH' % podman_exec)
    local_cmd = [podman_cmd]
    if self.get_option('podman_extra_args'):
        local_cmd += shlex.split(to_native(self.get_option('podman_extra_args'), errors='surrogate_or_strict'))
    if isinstance(cmd, str):
        local_cmd.append(cmd)
    else:
        local_cmd.extend(cmd)
    if use_container_id:
        local_cmd.append(self._container_id)
    if cmd_args:
        local_cmd += cmd_args
    local_cmd = [to_bytes(i, errors='surrogate_or_strict') for i in local_cmd]
    display.vvv('RUN %s' % (local_cmd,), host=self._container_id)
    p = subprocess.Popen(local_cmd, shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate(input=in_data)
    display.vvvvv('STDOUT %s' % stdout)
    display.vvvvv('STDERR %s' % stderr)
    display.vvvvv('RC CODE %s' % p.returncode)
    stdout = to_bytes(stdout, errors='surrogate_or_strict')
    stderr = to_bytes(stderr, errors='surrogate_or_strict')
    return (p.returncode, stdout, stderr)
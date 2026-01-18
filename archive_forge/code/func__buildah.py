from __future__ import (absolute_import, division, print_function)
import os
import shlex
import shutil
import subprocess
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_bytes, to_native, to_text
from ansible.plugins.connection import ConnectionBase, ensure_connect
from ansible.utils.display import Display
def _buildah(self, cmd, cmd_args=None, in_data=None, outfile_stdout=None):
    """
        run buildah executable

        :param cmd: buildah's command to execute (str)
        :param cmd_args: list of arguments to pass to the command (list of str/bytes)
        :param in_data: data passed to buildah's stdin
        :param outfile_stdout: file for writing STDOUT to
        :return: return code, stdout, stderr
        """
    buildah_exec = 'buildah'
    local_cmd = [buildah_exec]
    if isinstance(cmd, str):
        local_cmd.append(cmd)
    else:
        local_cmd.extend(cmd)
    if self.user and self.user != 'root':
        if cmd == 'run':
            local_cmd.extend(('--user', self.user))
        elif cmd == 'copy':
            local_cmd.extend(('--chown', self.user))
    local_cmd.append(self._container_id)
    if cmd_args:
        if isinstance(cmd_args, str):
            local_cmd.append(cmd_args)
        else:
            local_cmd.extend(cmd_args)
    local_cmd = [to_bytes(i, errors='surrogate_or_strict') for i in local_cmd]
    display.vvv('RUN %s' % (local_cmd,), host=self._container_id)
    if outfile_stdout:
        stdout_fd = open(outfile_stdout, 'wb')
    else:
        stdout_fd = subprocess.PIPE
    p = subprocess.Popen(local_cmd, shell=False, stdin=subprocess.PIPE, stdout=stdout_fd, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate(input=in_data)
    display.vvvv('STDOUT %s' % to_text(stdout))
    display.vvvv('STDERR %s' % to_text(stderr))
    display.vvvv('RC CODE %s' % p.returncode)
    stdout = to_bytes(stdout, errors='surrogate_or_strict')
    stderr = to_bytes(stderr, errors='surrogate_or_strict')
    return (p.returncode, stdout, stderr)
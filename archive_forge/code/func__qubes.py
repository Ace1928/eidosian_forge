from __future__ import (absolute_import, division, print_function)
import subprocess
from ansible.module_utils.common.text.converters import to_bytes
from ansible.plugins.connection import ConnectionBase, ensure_connect
from ansible.errors import AnsibleConnectionFailure
from ansible.utils.display import Display
def _qubes(self, cmd=None, in_data=None, shell='qubes.VMShell'):
    """run qvm-run executable

        :param cmd: cmd string for remote system
        :param in_data: data passed to qvm-run-vm's stdin
        :return: return code, stdout, stderr
        """
    display.vvvv('CMD: ', cmd)
    if not cmd.endswith('\n'):
        cmd = cmd + '\n'
    local_cmd = []
    local_cmd.extend(['qvm-run', '--pass-io', '--service'])
    if self.user != 'user':
        local_cmd.extend(['-u', self.user])
    local_cmd.append(self._remote_vmname)
    local_cmd.append(shell)
    local_cmd = [to_bytes(i, errors='surrogate_or_strict') for i in local_cmd]
    display.vvvv('Local cmd: ', local_cmd)
    display.vvv('RUN %s' % (local_cmd,), host=self._remote_vmname)
    p = subprocess.Popen(local_cmd, shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.stdin.write(to_bytes(cmd, errors='surrogate_or_strict'))
    stdout, stderr = p.communicate(input=in_data)
    return (p.returncode, stdout, stderr)
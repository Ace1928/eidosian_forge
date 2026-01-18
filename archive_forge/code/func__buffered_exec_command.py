from __future__ import (absolute_import, division, print_function)
import os
import os.path
import subprocess
import traceback
from ansible.errors import AnsibleError
from ansible.module_utils.six.moves import shlex_quote
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.text.converters import to_bytes
from ansible.plugins.connection import ConnectionBase, BUFSIZE
from ansible.utils.display import Display
def _buffered_exec_command(self, cmd, stdin=subprocess.PIPE):
    """ run a command on the zone.  This is only needed for implementing
        put_file() get_file() so that we don't have to read the whole file
        into memory.

        compared to exec_command() it looses some niceties like being able to
        return the process's exit code immediately.
        """
    local_cmd = [self.zlogin_cmd, self.zone, cmd]
    local_cmd = map(to_bytes, local_cmd)
    display.vvv('EXEC %s' % local_cmd, host=self.zone)
    p = subprocess.Popen(local_cmd, shell=False, stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p
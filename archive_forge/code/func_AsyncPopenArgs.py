from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import platform
import re
import subprocess
import sys
from googlecloudsdk.core.util import encoding
def AsyncPopenArgs(self):
    """Returns the args for spawning an async process using Popen on this OS.

    Make sure the main process does not wait for the new process. On windows
    this means setting the 0x8 creation flag to detach the process.

    Killing a group leader kills the whole group. Setting creation flag 0x200 on
    Windows or running setsid on *nix makes sure the new process is in a new
    session with the new process the group leader. This means it can't be killed
    if the parent is killed.

    Finally, all file descriptors (FD) need to be closed so that waiting for the
    output of the main process does not inadvertently wait for the output of the
    new process, which means waiting for the termination of the new process.
    If the new process wants to write to a file, it can open new FDs.

    Returns:
      {str:}, The args for spawning an async process using Popen on this OS.
    """
    args = {}
    if self.operating_system == OperatingSystem.WINDOWS:
        args['close_fds'] = True
        detached_process = 8
        create_new_process_group = 512
        args['creationflags'] = detached_process | create_new_process_group
    else:
        if sys.version_info[0] == 3 and sys.version_info[1] > 8:
            args['start_new_session'] = True
        else:
            args['preexec_fn'] = os.setsid
        args['close_fds'] = True
        args['stdin'] = subprocess.PIPE
        args['stdout'] = subprocess.PIPE
        args['stderr'] = subprocess.PIPE
    return args
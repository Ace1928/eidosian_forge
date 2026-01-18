from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import errno
import os
import re
import signal
import subprocess
import sys
import threading
import time
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import parallel
from googlecloudsdk.core.util import platforms
import six
from six.moves import map
def KillSubprocess(p):
    """Kills a subprocess using an OS specific method when python can't do it.

  This also kills all processes rooted in this process.

  Args:
    p: the Popen or multiprocessing.Process object to kill

  Raises:
    RuntimeError: if it fails to kill the process
  """
    code = None
    if hasattr(p, 'returncode'):
        code = p.returncode
    elif hasattr(p, 'exitcode'):
        code = p.exitcode
    if code is not None:
        return
    if platforms.OperatingSystem.Current() == platforms.OperatingSystem.WINDOWS:
        taskkill_process = subprocess.Popen(['taskkill', '/F', '/T', '/PID', six.text_type(p.pid)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = taskkill_process.communicate()
        if taskkill_process.returncode != 0 and _IsTaskKillError(stderr):
            raise RuntimeError('Failed to call taskkill on pid {0}\nstdout: {1}\nstderr: {2}'.format(p.pid, stdout, stderr))
    else:
        new_env = encoding.EncodeEnv(dict(os.environ))
        new_env['LANG'] = 'en_US.UTF-8'
        get_pids_process = subprocess.Popen(['ps', '-e', '-o', 'ppid=', '-o', 'pid='], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=new_env)
        stdout, stderr = get_pids_process.communicate()
        stdout = stdout.decode('utf-8')
        if get_pids_process.returncode != 0:
            raise RuntimeError('Failed to get subprocesses of process: {0}'.format(p.pid))
        pid_map = {}
        for line in stdout.strip().split('\n'):
            ppid, pid = re.match('\\s*(\\d+)\\s+(\\d+)', line).groups()
            ppid = int(ppid)
            pid = int(pid)
            children = pid_map.get(ppid)
            if not children:
                pid_map[ppid] = [pid]
            else:
                children.append(pid)
        all_pids = [p.pid]
        to_process = [p.pid]
        while to_process:
            current = to_process.pop()
            children = pid_map.get(current)
            if children:
                to_process.extend(children)
                all_pids.extend(children)
        for pid in all_pids:
            _KillPID(pid)
import errno
import functools
import os
import socket
import subprocess
import sys
from collections import namedtuple
from socket import AF_INET
from . import _common
from . import _psposix
from . import _psutil_posix as cext_posix
from . import _psutil_sunos as cext
from ._common import AF_INET6
from ._common import AccessDenied
from ._common import NoSuchProcess
from ._common import ZombieProcess
from ._common import debug
from ._common import get_procfs_path
from ._common import isfile_strict
from ._common import memoize_when_activated
from ._common import sockfam_to_enum
from ._common import socktype_to_enum
from ._common import usage_percent
from ._compat import PY3
from ._compat import FileNotFoundError
from ._compat import PermissionError
from ._compat import ProcessLookupError
from ._compat import b
def _get_unix_sockets(self, pid):
    """Get UNIX sockets used by process by parsing 'pfiles' output."""
    cmd = ['pfiles', str(pid)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    if PY3:
        stdout, stderr = (x.decode(sys.stdout.encoding) for x in (stdout, stderr))
    if p.returncode != 0:
        if 'permission denied' in stderr.lower():
            raise AccessDenied(self.pid, self._name)
        if 'no such process' in stderr.lower():
            raise NoSuchProcess(self.pid, self._name)
        raise RuntimeError('%r command error\n%s' % (cmd, stderr))
    lines = stdout.split('\n')[2:]
    for i, line in enumerate(lines):
        line = line.lstrip()
        if line.startswith('sockname: AF_UNIX'):
            path = line.split(' ', 2)[2]
            type = lines[i - 2].strip()
            if type == 'SOCK_STREAM':
                type = socket.SOCK_STREAM
            elif type == 'SOCK_DGRAM':
                type = socket.SOCK_DGRAM
            else:
                type = -1
            yield (-1, socket.AF_UNIX, type, path, '', _common.CONN_NONE)
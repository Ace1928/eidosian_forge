from __future__ import annotations
import errno
import gc
import io
import os
import signal
import stat
import sys
import traceback
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from zope.interface import implementer
from twisted.internet import abstract, error, fdesc
from twisted.internet._baseprocess import BaseProcess
from twisted.internet.interfaces import IProcessTransport
from twisted.internet.main import CONNECTION_DONE, CONNECTION_LOST
from twisted.python import failure, log
from twisted.python.runtime import platform
from twisted.python.util import switchUID
def _trySpawnInsteadOfFork(self, path, uid, gid, executable, args, environment, kwargs):
    """
        Try to use posix_spawnp() instead of fork(), if possible.

        @return: a boolean indicating whether posix_spawnp() was used or not.
        """
    if uid is not None or gid is not None or (path is not None and os.path.abspath(path) != os.path.abspath('.')) or getattr(self._reactor, '_neverUseSpawn', False):
        return False
    fdmap = kwargs.get('fdmap')
    fdState = []
    for eachFD in _listOpenFDs():
        try:
            isCloseOnExec = fcntl.fcntl(eachFD, fcntl.F_GETFD, fcntl.FD_CLOEXEC)
        except OSError:
            pass
        else:
            fdState.append((eachFD, isCloseOnExec))
    if environment is None:
        environment = os.environ
    setSigDef = [everySignal for everySignal in range(1, signal.NSIG) if signal.getsignal(everySignal) == signal.SIG_IGN]
    self.pid = os.posix_spawnp(executable, args, environment, file_actions=_getFileActions(fdState, fdmap, doClose=_PS_CLOSE, doDup2=_PS_DUP2), setsigdef=setSigDef)
    self.status = -1
    return True
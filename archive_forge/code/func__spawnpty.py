import os
import sys
import time
import pty
import tty
import errno
import signal
from contextlib import contextmanager
import ptyprocess
from ptyprocess.ptyprocess import use_native_pty_fork
from .exceptions import ExceptionPexpect, EOF, TIMEOUT
from .spawnbase import SpawnBase
from .utils import (
def _spawnpty(self, args, **kwargs):
    """Spawn a pty and return an instance of PtyProcess."""
    return ptyprocess.PtyProcess.spawn(args, **kwargs)
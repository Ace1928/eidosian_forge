import errno
import getpass
import logging
import os
import socket
import subprocess
import sys
from binascii import hexlify
from typing import Dict, Optional, Set, Tuple, Type
from .. import bedding, config, errors, osutils, trace, ui
import weakref
def _close_ssh_proc(proc, sock):
    """Carefully close stdin/stdout and reap the SSH process.

    If the pipes are already closed and/or the process has already been
    wait()ed on, that's ok, and no error is raised.  The goal is to do our best
    to clean up (whether or not a clean up was already tried).
    """
    funcs = []
    for closeable in (proc.stdin, proc.stdout, sock):
        if closeable is not None:
            funcs.append(closeable.close)
    funcs.append(proc.wait)
    for func in funcs:
        try:
            func()
        except OSError:
            continue
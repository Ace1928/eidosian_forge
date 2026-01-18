from __future__ import division
from __future__ import print_function
import collections
import contextlib
import errno
import functools
import os
import socket
import stat
import sys
import threading
import warnings
from collections import namedtuple
from socket import AF_INET
from socket import SOCK_DGRAM
from socket import SOCK_STREAM
class ZombieProcess(NoSuchProcess):
    """Exception raised when querying a zombie process. This is
    raised on macOS, BSD and Solaris only, and not always: depending
    on the query the OS may be able to succeed anyway.
    On Linux all zombie processes are querable (hence this is never
    raised). Windows doesn't have zombie processes.
    """
    __module__ = 'psutil'

    def __init__(self, pid, name=None, ppid=None, msg=None):
        NoSuchProcess.__init__(self, pid, name, msg)
        self.ppid = ppid
        self.msg = msg or "PID still exists but it's a zombie"
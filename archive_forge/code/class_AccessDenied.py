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
class AccessDenied(Error):
    """Exception raised when permission to perform an action is denied."""
    __module__ = 'psutil'

    def __init__(self, pid=None, name=None, msg=None):
        Error.__init__(self)
        self.pid = pid
        self.name = name
        self.msg = msg or ''
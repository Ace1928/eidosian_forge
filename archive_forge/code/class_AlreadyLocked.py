from __future__ import absolute_import
import functools
import os
import socket
import threading
import warnings
class AlreadyLocked(LockError):
    """Some other thread/process is locking the file.

    >>> try:
    ...   raise AlreadyLocked
    ... except LockError:
    ...   pass
    """
    pass
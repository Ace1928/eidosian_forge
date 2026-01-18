from __future__ import absolute_import
import functools
import os
import socket
import threading
import warnings
class NotMyLock(UnlockError):
    """Raised when an attempt is made to unlock a file someone else locked.

    >>> try:
    ...   raise NotMyLock
    ... except UnlockError:
    ...   pass
    """
    pass
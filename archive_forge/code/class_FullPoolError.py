from __future__ import annotations
import socket
import typing
import warnings
from email.errors import MessageDefect
from http.client import IncompleteRead as httplib_IncompleteRead
class FullPoolError(PoolError):
    """Raised when we try to add a connection to a full pool in blocking mode."""
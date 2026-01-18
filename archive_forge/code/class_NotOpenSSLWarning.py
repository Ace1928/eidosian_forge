from __future__ import annotations
import socket
import typing
import warnings
from email.errors import MessageDefect
from http.client import IncompleteRead as httplib_IncompleteRead
class NotOpenSSLWarning(SecurityWarning):
    """Warned when using unsupported SSL library"""
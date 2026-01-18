import os
import time
import calendar
import socket
import errno
import copy
import warnings
import email
import email.message
import email.generator
import io
import contextlib
from types import GenericAlias
def _sync_close(f):
    """Close file f, ensuring all changes are physically on disk."""
    _sync_flush(f)
    f.close()
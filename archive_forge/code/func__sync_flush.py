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
def _sync_flush(f):
    """Ensure changes to file f are physically on disk."""
    f.flush()
    if hasattr(os, 'fsync'):
        os.fsync(f.fileno())
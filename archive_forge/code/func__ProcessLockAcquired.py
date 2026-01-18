from __future__ import print_function
import argparse
import contextlib
import datetime
import json
import os
import threading
import warnings
import httplib2
import oauth2client
import oauth2client.client
from oauth2client import service_account
from oauth2client import tools  # for gflags declarations
import six
from six.moves import http_client
from six.moves import urllib
from apitools.base.py import exceptions
from apitools.base.py import util
@contextlib.contextmanager
def _ProcessLockAcquired(self):
    """Context manager for process locks with timeout."""
    try:
        is_locked = self._process_lock.acquire(timeout=self._lock_timeout)
        yield is_locked
    finally:
        if is_locked:
            self._process_lock.release()
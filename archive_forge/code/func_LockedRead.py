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
def LockedRead(self):
    """Acquire an interprocess lock and dump cache contents.

        This method safely acquires the locks then reads a string
        from the cache file. If the file does not exist and cannot
        be created, it will return None. If the locks cannot be
        acquired, this will also return None.

        Returns:
          cache data - string if present, None on failure.
        """
    file_contents = None
    with self._thread_lock:
        if not self._EnsureFileExists():
            return None
        with self._process_lock_getter() as acquired_plock:
            if not acquired_plock:
                return None
            with open(self._filename, 'rb') as f:
                file_contents = f.read().decode(encoding=self._encoding)
    return file_contents
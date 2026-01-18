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
def LockedWrite(self, cache_data):
    """Acquire an interprocess lock and write a string.

        This method safely acquires the locks then writes a string
        to the cache file. If the string is written successfully
        the function will return True, if the write fails for any
        reason it will return False.

        Args:
          cache_data: string or bytes to write.

        Returns:
          bool: success
        """
    if isinstance(cache_data, six.text_type):
        cache_data = cache_data.encode(encoding=self._encoding)
    with self._thread_lock:
        if not self._EnsureFileExists():
            return False
        with self._process_lock_getter() as acquired_plock:
            if not acquired_plock:
                return False
            with open(self._filename, 'wb') as f:
                f.write(cache_data)
            return True
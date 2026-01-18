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
def _CheckCacheFileForMatch(self, cache_filename, scopes):
    """Checks the cache file to see if it matches the given credentials.

        Args:
          cache_filename: Cache filename to check.
          scopes: Scopes for the desired credentials.

        Returns:
          List of scopes (if cache matches) or None.
        """
    creds = {'scopes': sorted(list(scopes)) if scopes else None, 'svc_acct_name': self.__service_account_name}
    cache_file = _MultiProcessCacheFile(cache_filename)
    try:
        cached_creds_str = cache_file.LockedRead()
        if not cached_creds_str:
            return None
        cached_creds = json.loads(cached_creds_str)
        if creds['svc_acct_name'] == cached_creds['svc_acct_name']:
            if creds['scopes'] in (None, cached_creds['scopes']):
                return cached_creds['scopes']
    except KeyboardInterrupt:
        raise
    except:
        pass
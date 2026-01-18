import json
import os.path
import shutil
import tempfile
import unittest
import mock
import six
from apitools.base.py import credentials_lib
from apitools.base.py import util
def _RunGceAssertionCredentials(self, service_account_name=None, scopes=None, cache_filename=None):
    kwargs = {}
    if service_account_name is not None:
        kwargs['service_account_name'] = service_account_name
    if cache_filename is not None:
        kwargs['cache_filename'] = cache_filename
    service_account_name = service_account_name or 'default'
    credentials = credentials_lib.GceAssertionCredentials(scopes, **kwargs)
    self.assertIsNone(credentials._refresh(None))
    return credentials
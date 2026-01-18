from __future__ import absolute_import
import logging
import os
import warnings
import six
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
from google.oauth2 import service_account
def _make_default_http():
    if certifi is not None:
        return urllib3.PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())
    else:
        return urllib3.PoolManager()
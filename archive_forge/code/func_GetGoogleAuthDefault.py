from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import base64
import collections
import copy
import datetime
import enum
import hashlib
import json
import os
import sqlite3
from google.auth import compute_engine as google_auth_compute_engine
from google.auth import credentials as google_auth_creds
from google.auth import exceptions as google_auth_exceptions
from google.auth import external_account as google_auth_external_account
from google.auth import external_account_authorized_user as google_auth_external_account_authorized_user
from google.auth import impersonated_credentials as google_auth_impersonated
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import devshell as c_devshell
from googlecloudsdk.core.credentials import exceptions as c_exceptions
from googlecloudsdk.core.credentials import introspect as c_introspect
from googlecloudsdk.core.util import files
from oauth2client import client
from oauth2client import service_account
from oauth2client.contrib import gce as oauth2client_gce
import six
def GetGoogleAuthDefault():
    """Get the google.auth._default module.

  All messages from logging and warnings are muted because they are for
  ADC consumers (client libraries). The message are irrelevant and confusing to
  gcloud auth application-default users. gcloud auth application-default
  are the ADC producer.

  Returns:
    The google.auth._default module with logging/warnings muted.
  """
    global GOOGLE_AUTH_DEFAULT
    if GOOGLE_AUTH_DEFAULT:
        return GOOGLE_AUTH_DEFAULT
    from google.auth import _default
    import warnings
    warnings.filterwarnings('ignore', category=Warning, module='google.auth._default')
    _default._LOGGER.setLevel(VERBOSITY_MUTED)
    GOOGLE_AUTH_DEFAULT = _default
    return GetGoogleAuthDefault()
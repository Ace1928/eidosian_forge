import codecs
import copy
import http.client
import json
import logging
import os
import pkgutil
import platform
import sys
import textwrap
import time
import traceback
from typing import Any, Dict, List, Optional, TextIO
from absl import app
from absl import flags
from google.auth import version as google_auth_version
from google.oauth2 import credentials as google_oauth2
import googleapiclient
import httplib2
import oauth2client_4_0.client
import requests
import urllib3
from utils import bq_error
from utils import bq_logging
from pyglib import stringutil
def GetSanitizedCredentialForDiscoveryRequest(use_google_auth: bool, credentials: Any) -> Any:
    """Return the sanitized input credentials used to make discovery requests.

  When the credentials object is not Google Auth, return the original
  credentials. When it's of type google.oauth2.Credentials, return a copy of the
  original credentials without quota project ID. The returned credentials object
  is used in bigquery_client to construct an http object for discovery requests.

  Args:
    use_google_auth: True if Google Auth credentials should be used.
    credentials: The credentials object.
  """
    if use_google_auth and isinstance(credentials, google_oauth2.Credentials):
        sanitized_credentials = copy.deepcopy(credentials)
        sanitized_credentials._quota_project_id = None
        return sanitized_credentials
    return credentials
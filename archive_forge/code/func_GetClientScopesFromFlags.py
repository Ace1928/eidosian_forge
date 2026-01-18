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
def GetClientScopesFromFlags() -> List[str]:
    """Returns auth scopes based on user supplied flags."""
    client_scope = [_BIGQUERY_SCOPE, _CLOUD_PLATFORM_SCOPE]
    if FLAGS.enable_gdrive:
        client_scope.append(_GDRIVE_SCOPE)
    client_scope.append(_REAUTH_SCOPE)
    return client_scope
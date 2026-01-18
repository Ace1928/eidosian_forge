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
def _GenerateContactUsMessage() -> str:
    """Generates the Contact Us message."""
    contact_us_msg = 'Please file a bug report in our public issue tracker:\n  https://issuetracker.google.com/issues/new?component=187149&template=0\nPlease include a brief description of the steps that led to this issue, as well as any rows that can be made public from the following information: \n\n'
    try:
        gcloud_properties_file = GetGcloudConfigFilename()
        gcloud_core_properties = _ProcessConfigSection(gcloud_properties_file, 'core')
        if 'account' in gcloud_core_properties and '@google.com' in gcloud_core_properties['account']:
            contact_us_msg = contact_us_msg.replace('public', 'internal').replace('https://issuetracker.google.com/issues/new?component=187149&template=0', 'http://b/issues/new?component=60322&template=178900')
    except Exception:
        pass
    return contact_us_msg
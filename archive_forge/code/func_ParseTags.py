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
def ParseTags(tags: str) -> Dict[str, str]:
    """Parses user-supplied string representing tags.

  Args:
    tags: A comma separated user-supplied string representing tags. It is
      expected to be in the format "key1:value1,key2:value2".

  Returns:
    A dictionary mapping tag keys to tag values.

  Raises:
    UsageError: Incorrect tags or no tags are supplied.
  """
    tags = tags.strip()
    if not tags:
        raise app.UsageError('No tags supplied')
    tags_dict = {}
    for key_value in tags.split(','):
        k, _, v = key_value.partition(':')
        k = k.strip()
        if not k:
            raise app.UsageError('Tag key cannot be None')
        v = v.strip()
        if not v:
            raise app.UsageError('Tag value cannot be None')
        if k in tags_dict:
            raise app.UsageError('Cannot specify tag key "%s" multiple times' % k)
        tags_dict[k] = v
    return tags_dict
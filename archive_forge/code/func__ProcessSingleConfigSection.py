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
def _ProcessSingleConfigSection(file: TextIO, section_name: str) -> Dict[str, str]:
    """Read a configuration file section returned as a dictionary.

  Args:
    file: The opened configuration file object.
    section_name: Name of the section to read.

  Returns:
    A dictionary of flag names and values from that section of the file.
  """
    dictionary = {}
    in_section = not section_name
    for line in file:
        if line.lstrip().startswith('[') and line.rstrip().endswith(']'):
            next_section = line.strip()[1:-1]
            in_section = section_name == next_section
            continue
        elif not in_section:
            continue
        elif line.lstrip().startswith('#') or not line.strip():
            continue
        flag, equalsign, value = line.partition('=')
        if not equalsign:
            value = 'true'
        flag = flag.strip()
        value = value.strip()
        while flag.startswith('-'):
            flag = flag[1:]
        dictionary[flag] = value
    return dictionary
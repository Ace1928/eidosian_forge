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
def _ProcessConfigSections(filename: str, section_names: List[str]) -> Dict[str, Dict[str, str]]:
    """Read configuration file sections returned as a nested dictionary.

  Args:
    filename: The filename of the configuration file.
    section_names: A list of the section names.

  Returns:
    A nested dictionary of section names to flag names and values from the file.
  """
    dictionary = {}
    if not os.path.exists(filename):
        logging.debug('File not found: %s', filename)
        return dictionary
    try:
        with open(filename) as rcfile:
            for section_name in section_names:
                dictionary[section_name] = _ProcessSingleConfigSection(rcfile, section_name)
                rcfile.seek(0)
    except IOError as e:
        logging.debug('IOError opening config file %s: %s', filename, e)
    return dictionary
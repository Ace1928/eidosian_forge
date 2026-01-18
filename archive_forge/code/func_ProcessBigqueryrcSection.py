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
def ProcessBigqueryrcSection(section_name: Optional[str], flag_values) -> None:
    """Read the bigqueryrc file into flag_values for section section_name.

  Args:
    section_name: if None, read the global flag settings.
    flag_values: FLAGS instance.

  Raises:
    UsageError: Unknown flag found.
  """
    bigqueryrc = GetBigqueryRcFilename()
    dictionary = _ProcessConfigSection(filename=bigqueryrc, section_name=section_name)
    for flag, value in dictionary.items():
        if flag not in flag_values:
            raise app.UsageError('Unknown flag %s found in bigqueryrc file in section %s' % (flag, section_name if section_name else 'global'))
        if not flag_values[flag].present:
            UpdateFlag(flag_values, flag, value)
        else:
            flag_type = flag_values[flag].flag_type()
            if flag_type.startswith('multi'):
                old_value = getattr(flag_values, flag)
                flag_values[flag].parse(value)
                setattr(flag_values, flag, old_value + getattr(flag_values, flag))
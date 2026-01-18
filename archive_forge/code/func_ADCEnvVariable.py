from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import logging
import os
import sqlite3
import time
from typing import Dict
import uuid
import googlecloudsdk
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import pkg_resources
from googlecloudsdk.core.util import platforms
import six
def ADCEnvVariable():
    """Gets the value of the ADC environment variable.

  Returns:
    str, The value of the env var or None if unset.
  """
    from google.auth import environment_vars
    return encoding.GetEncodedValue(os.environ, environment_vars.CREDENTIALS, None)
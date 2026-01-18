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
def GetCID():
    """Gets the client id from the config file, or generates a new one.

  Returns:
    str, The hex string of the client id.
  """
    uuid_path = Paths().cid_path
    try:
        cid = file_utils.ReadFileContents(uuid_path)
        if cid:
            return cid
    except file_utils.Error:
        pass
    return _GenerateCID(uuid_path)
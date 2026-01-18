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
def LegacyCredentialsGSUtilPath(self, account):
    """Gets the path to store legacy gsutil credentials in.

    Args:
      account: str, Email account tied to the authorizing credentials.

    Returns:
      str, The path to the gsutil credentials file.
    """
    return os.path.join(self.LegacyCredentialsDir(account), '.boto')
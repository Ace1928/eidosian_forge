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
def EnsureSDKWriteAccess(sdk_root_override=None):
    """Error if the current user does not have write access to the sdk root.

  Args:
    sdk_root_override: str, The full path to the sdk root to use instead of
      using config.Paths().sdk_root.

  Raises:
    exceptions.RequiresAdminRightsError: If the sdk root is defined and the user
      does not have write access.
  """
    sdk_root = sdk_root_override or Paths().sdk_root
    if sdk_root and (not file_utils.HasWriteAccessInDir(sdk_root)):
        raise exceptions.RequiresAdminRightsError(sdk_root)
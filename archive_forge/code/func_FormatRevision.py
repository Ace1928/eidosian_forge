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
@staticmethod
def FormatRevision(time_struct):
    """Formats a given time as a revision string for a component snapshot.

    Args:
      time_struct: time.struct_time, The time you want to format.

    Returns:
      int, A revision number from a component snapshot.  This is a int but
      formatted as an actual date in seconds (i.e 20151009132504).  It is *NOT*
      seconds since the epoch.
    """
    return int(time.strftime(InstallationConfig.REVISION_FORMAT_STRING, time_struct))
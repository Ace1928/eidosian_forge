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
def ParseRevisionAsSeconds(revision):
    """Parse the given revision into seconds since the epoch.

    Args:
      revision: long, A revision number from a component snapshot.  This is a
        long int but formatted as an actual date in seconds (i.e
        20151009132504). It is *NOT* seconds since the epoch.

    Returns:
      int, The number of seconds since the epoch that this revision represents.
    """
    return time.mktime(InstallationConfig.ParseRevision(revision))
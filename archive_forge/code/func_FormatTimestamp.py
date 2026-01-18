from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import re
from typing import Any
from googlecloudsdk.api_lib.functions.v1 import util as api_util_v1
from googlecloudsdk.api_lib.functions.v2 import client as client_v2
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.functions import flags
import six
def FormatTimestamp(timestamp):
    """Formats a timestamp which will be presented to a user.

  Args:
    timestamp: Raw timestamp string in RFC3339 UTC "Zulu" format.

  Returns:
    Formatted timestamp string.
  """
    return re.sub('(\\.\\d{3})\\d*Z$', '\\1', timestamp.replace('T', ' '))
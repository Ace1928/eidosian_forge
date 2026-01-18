from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from calendar import timegm
import getpass
import logging
import os
import re
import time
import six
from gslib.exception import CommandException
from gslib.tz_utc import UTC
from gslib.utils.metadata_util import CreateCustomMetadata
from gslib.utils.metadata_util import GetValueFromObjectCustomMetadata
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils.unit_util import SECONDS_PER_DAY
def ValidatePOSIXMode(mode):
    """Validates whether the mode is valid.

  In order for the mode to be valid either the user, group, or other byte must
  be >= 4.

  Args:
    mode: The mode as a 3-digit, base-8 integer.

  Returns:
    True/False
  """
    return MODE_REGEX.match(oct(mode)[-3:]) and (mode & U_R or mode & G_R or mode & O_R)
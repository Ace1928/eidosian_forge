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
def ConvertModeToBase8(mode):
    """Converts a base-10 mode integer from os.stat to base-8."""
    return int(oct(mode)[-3:])
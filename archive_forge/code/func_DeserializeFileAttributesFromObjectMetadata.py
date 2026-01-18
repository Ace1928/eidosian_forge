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
def DeserializeFileAttributesFromObjectMetadata(obj_metadata, url_str):
    """Parses the POSIX attributes from the supplied metadata.

  Args:
    obj_metadata: The metadata for an object.
    url_str: File/object path that provides context if a warning is thrown.

  Returns:
    A POSIXAttribute object with the retrieved values or a default value for
    any attribute that could not be found.
  """
    posix_attrs = POSIXAttributes()
    found, atime = GetValueFromObjectCustomMetadata(obj_metadata, ATIME_ATTR, NA_TIME)
    try:
        atime = long(atime)
        if found and atime <= NA_TIME:
            WarnNegativeAttribute('atime', url_str)
            atime = NA_TIME
        elif atime > long(time.time()) + SECONDS_PER_DAY:
            WarnFutureTimestamp('atime', url_str)
            atime = NA_TIME
    except ValueError:
        WarnInvalidValue('atime', url_str)
        atime = NA_TIME
    posix_attrs.atime = atime
    DeserializeIDAttribute(obj_metadata, GID_ATTR, url_str, posix_attrs)
    DeserializeIDAttribute(obj_metadata, UID_ATTR, url_str, posix_attrs)
    found, mode = GetValueFromObjectCustomMetadata(obj_metadata, MODE_ATTR, NA_MODE)
    if found and MODE_REGEX.match(mode):
        try:
            posix_attrs.mode = POSIXMode(int(mode))
        except ValueError:
            WarnInvalidValue('mode', url_str)
    return posix_attrs
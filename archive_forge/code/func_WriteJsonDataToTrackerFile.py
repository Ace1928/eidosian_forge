from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import errno
import hashlib
import json
import os
import re
import sys
import six
from boto import config
from gslib.exception import CommandException
from gslib.utils.boto_util import GetGsutilStateDir
from gslib.utils.boto_util import ResumableThreshold
from gslib.utils.constants import UTF8
from gslib.utils.hashing_helper import GetMd5
from gslib.utils.system_util import CreateDirIfNeeded
def WriteJsonDataToTrackerFile(tracker_file_name, data):
    """Create a tracker file and write json data to it.

  Raises:
    TypeError: If the data is not JSON serializable
  """
    try:
        json_str = json.dumps(data)
    except TypeError as err:
        raise RaiseUnwritableTrackerFileException(tracker_file_name, err.strerror)
    _WriteTrackerFile(tracker_file_name, json_str)
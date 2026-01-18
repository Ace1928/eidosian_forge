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
def _WriteTrackerFile(tracker_file_name, data):
    """Creates a tracker file, storing the input data."""
    try:
        fd = os.open(tracker_file_name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 384)
        with os.fdopen(fd, 'w') as tf:
            tf.write(data)
        return False
    except (IOError, OSError) as e:
        raise RaiseUnwritableTrackerFileException(tracker_file_name, e.strerror)
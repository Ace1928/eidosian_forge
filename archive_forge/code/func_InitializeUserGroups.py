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
def InitializeUserGroups():
    """Initializes the set of groups that the user is in.

  Should only be called if the flag for preserving POSIX attributes is set.
  """
    global USER_GROUPS
    if IS_WINDOWS:
        return
    user_id = os.getuid()
    user_name = pwd.getpwuid(user_id).pw_name
    USER_GROUPS = set([pwd.getpwuid(user_id).pw_gid] + [g.gr_gid for g in grp.getgrall() if user_name in g.gr_mem])
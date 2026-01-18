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
def NeedsPOSIXAttributeUpdate(src_atime, dst_atime, src_mtime, dst_mtime, src_uid, dst_uid, src_gid, dst_gid, src_mode, dst_mode):
    """Checks whether an update for any POSIX attribute is needed.

  Args:
    src_atime: The source access time.
    dst_atime: The destination access time.
    src_mtime: The source modification time.
    dst_mtime: The destination modification time.
    src_uid: The source user ID.
    dst_uid: The destination user ID.
    src_gid: The source group ID.
    dst_gid: The destination group ID.
    src_mode: The source mode.
    dst_mode: The destination mode.

  Returns:
    A tuple containing a POSIXAttribute object and a boolean for whether an
    update was needed.
  """
    posix_attrs = POSIXAttributes()
    has_src_atime = src_atime > NA_TIME
    has_dst_atime = dst_atime > NA_TIME
    has_src_mtime = src_mtime > NA_TIME
    has_dst_mtime = dst_mtime > NA_TIME
    has_src_uid = src_uid > NA_ID
    has_dst_uid = dst_uid > NA_ID
    has_src_gid = src_gid > NA_ID
    has_dst_gid = dst_gid > NA_ID
    has_src_mode = src_mode > NA_MODE
    has_dst_mode = dst_mode > NA_MODE
    if has_src_atime and (not has_dst_atime):
        posix_attrs.atime = src_atime
    if has_src_mtime and (not has_dst_mtime):
        posix_attrs.mtime = src_mtime
    if has_src_uid and (not has_dst_uid):
        posix_attrs.uid = src_uid
    if has_src_gid and (not has_dst_gid):
        posix_attrs.gid = src_gid
    if has_src_mode and (not has_dst_mode):
        posix_attrs.mode.permissions = src_mode
    return (posix_attrs, has_src_atime and (not has_dst_atime) or (has_src_mtime and (not has_dst_mtime)) or (has_src_uid and (not has_dst_uid)) or (has_src_gid and (not has_dst_gid)) or (has_src_mode and (not has_dst_mode)))
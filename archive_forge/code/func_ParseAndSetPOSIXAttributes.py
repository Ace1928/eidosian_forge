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
def ParseAndSetPOSIXAttributes(path, obj_metadata, is_rsync=False, preserve_posix=False):
    """Parses POSIX attributes from obj_metadata and sets them.

  Attributes will only be set if they exist in custom metadata. This function
  should only be called after ValidateFilePermissionAccess has been called for
  the specific file/object so as not to orphan files.

  Args:
    path: The local filesystem path for the file. Valid metadata attributes will
          be set for the file located at path, some attributes will only be set
          if preserve_posix is set to True.
    obj_metadata: The metadata for the file/object.
    is_rsync: Whether or not the caller is the rsync command. Used to determine
              if timeCreated should be used.
    preserve_posix: Whether or not all POSIX attributes should be set.
  """
    if obj_metadata is None:
        raise CommandException('obj_metadata cannot be None for %s' % path)
    try:
        found_at, atime = GetValueFromObjectCustomMetadata(obj_metadata, ATIME_ATTR, default_value=NA_TIME)
        found_mt, mtime = GetValueFromObjectCustomMetadata(obj_metadata, MTIME_ATTR, default_value=NA_TIME)
        found_uid, uid = GetValueFromObjectCustomMetadata(obj_metadata, UID_ATTR, default_value=NA_ID)
        found_gid, gid = GetValueFromObjectCustomMetadata(obj_metadata, GID_ATTR, default_value=NA_ID)
        found_mode, mode = GetValueFromObjectCustomMetadata(obj_metadata, MODE_ATTR, default_value=NA_MODE)
        if found_mt:
            mtime = long(mtime)
            if not preserve_posix:
                atime_tmp = os.stat(path).st_atime
                os.utime(path, (atime_tmp, mtime))
                return
        elif is_rsync:
            mtime = ConvertDatetimeToPOSIX(obj_metadata.timeCreated)
            os.utime(path, (mtime, mtime))
        if not preserve_posix:
            return
        if found_at:
            atime = long(atime)
        if atime > NA_TIME and mtime > NA_TIME:
            os.utime(path, (atime, mtime))
        elif atime > NA_TIME and mtime <= NA_TIME:
            mtime_tmp = os.stat(path).st_mtime
            os.utime(path, (atime, mtime_tmp))
        elif atime <= NA_TIME and mtime > NA_TIME:
            atime_tmp = os.stat(path).st_atime
            os.utime(path, (atime_tmp, mtime))
        if IS_WINDOWS:
            return
        if found_uid and os.geteuid() == 0:
            uid = int(uid)
        else:
            uid = NA_ID
        if found_gid:
            gid = int(gid)
        if uid > NA_ID and gid > NA_ID:
            os.chown(path, uid, gid)
        elif uid > NA_ID and gid <= NA_ID:
            os.chown(path, uid, -1)
        elif uid <= NA_ID and gid > NA_ID:
            os.chown(path, -1, gid)
        if found_mode:
            mode = int(str(mode), 8)
            os.chmod(path, mode)
    except ValueError:
        raise CommandException('Check POSIX attribute values for %s' % obj_metadata.name)
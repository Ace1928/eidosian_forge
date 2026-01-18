from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import collections
import errno
import heapq
import io
from itertools import islice
import logging
import os
import re
import tempfile
import textwrap
import time
import traceback
import sys
import six
from six.moves import urllib
from boto import config
import crcmod
from gslib.bucket_listing_ref import BucketListingObject
from gslib.cloud_api import NotFoundException
from gslib.cloud_api import ServiceException
from gslib.command import Command
from gslib.command import DummyArgChecker
from gslib.commands.cp import ShimTranslatePredefinedAclSubOptForCopy
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.metrics import LogPerformanceSummaryParams
from gslib.plurality_checkable_iterator import PluralityCheckableIterator
from gslib.seek_ahead_thread import SeekAheadResult
from gslib.sig_handling import GetCaughtSignals
from gslib.sig_handling import RegisterSignalHandler
from gslib.storage_url import GenerationFromUrlAndString
from gslib.storage_url import IsCloudSubdirPlaceholder
from gslib.storage_url import StorageUrlFromString
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import constants
from gslib.utils import copy_helper
from gslib.utils import parallelism_framework_util
from gslib.utils.boto_util import UsingCrcmodExtension
from gslib.utils.cloud_api_helper import GetCloudApiInstance
from gslib.utils.copy_helper import CreateCopyHelperOpts
from gslib.utils.copy_helper import GetSourceFieldsNeededForCopy
from gslib.utils.copy_helper import GZIP_ALL_FILES
from gslib.utils.copy_helper import SkipUnsupportedObjectError
from gslib.utils.hashing_helper import CalculateB64EncodedCrc32cFromContents
from gslib.utils.hashing_helper import CalculateB64EncodedMd5FromContents
from gslib.utils.hashing_helper import SLOW_CRCMOD_RSYNC_WARNING
from gslib.utils.hashing_helper import SLOW_CRCMOD_WARNING
from gslib.utils.metadata_util import CreateCustomMetadata
from gslib.utils.metadata_util import GetValueFromObjectCustomMetadata
from gslib.utils.metadata_util import ObjectIsGzipEncoded
from gslib.utils.posix_util import ATIME_ATTR
from gslib.utils.posix_util import ConvertDatetimeToPOSIX
from gslib.utils.posix_util import ConvertModeToBase8
from gslib.utils.posix_util import DeserializeFileAttributesFromObjectMetadata
from gslib.utils.posix_util import GID_ATTR
from gslib.utils.posix_util import InitializePreservePosixData
from gslib.utils.posix_util import MODE_ATTR
from gslib.utils.posix_util import MTIME_ATTR
from gslib.utils.posix_util import NA_ID
from gslib.utils.posix_util import NA_MODE
from gslib.utils.posix_util import NA_TIME
from gslib.utils.posix_util import NeedsPOSIXAttributeUpdate
from gslib.utils.posix_util import ParseAndSetPOSIXAttributes
from gslib.utils.posix_util import POSIXAttributes
from gslib.utils.posix_util import SerializeFileAttributesToObjectMetadata
from gslib.utils.posix_util import UID_ATTR
from gslib.utils.posix_util import ValidateFilePermissionAccess
from gslib.utils.posix_util import WarnFutureTimestamp
from gslib.utils.posix_util import WarnInvalidValue
from gslib.utils.posix_util import WarnNegativeAttribute
from gslib.utils.rsync_util import DiffAction
from gslib.utils.rsync_util import RsyncDiffToApply
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils.translation_helper import CopyCustomMetadata
from gslib.utils.unit_util import CalculateThroughput
from gslib.utils.unit_util import SECONDS_PER_DAY
from gslib.utils.unit_util import TEN_MIB
from gslib.wildcard_iterator import CreateWildcardIterator
def _BuildTmpOutputLine(blr):
    """Builds line to output to temp file for given BucketListingRef.

  Args:
    blr: The BucketListingRef.

  Returns:
    The output line, formatted as
    _EncodeUrl(URL)<sp>size<sp>time_created<sp>atime<sp>mtime<sp>mode<sp>uid<sp>
    gid<sp>crc32c<sp>md5 where md5 will only be present for cloud URLs that
    aren't composite objects. A missing field is populated with '-', or -1 in
    the case of atime/mtime/time_created.
  """
    atime = NA_TIME
    crc32c = _NA
    gid = NA_ID
    md5 = _NA
    mode = NA_MODE
    mtime = NA_TIME
    time_created = NA_TIME
    uid = NA_ID
    url = blr.storage_url
    if url.IsFileUrl():
        mode, _, _, _, uid, gid, size, atime, mtime, _ = os.stat(url.object_name)
        atime = long(atime)
        mtime = long(mtime)
        mode = ConvertModeToBase8(mode)
        if atime < 0:
            atime = NA_TIME
        if mtime < 0:
            mtime = NA_TIME
    elif url.IsCloudUrl():
        size = blr.root_object.size
        if blr.root_object.metadata is not None:
            found_m, mtime_str = GetValueFromObjectCustomMetadata(blr.root_object, MTIME_ATTR, NA_TIME)
            try:
                mtime = long(mtime_str)
                if found_m and mtime <= NA_TIME:
                    WarnNegativeAttribute('mtime', url.url_string)
                if mtime > long(time.time()) + SECONDS_PER_DAY:
                    WarnFutureTimestamp('mtime', url.url_string)
            except ValueError:
                WarnInvalidValue('mtime', url.url_string)
                mtime = NA_TIME
            posix_attrs = DeserializeFileAttributesFromObjectMetadata(blr.root_object, url.url_string)
            mode = posix_attrs.mode.permissions
            atime = posix_attrs.atime
            uid = posix_attrs.uid
            gid = posix_attrs.gid
        time_created = ConvertDatetimeToPOSIX(blr.root_object.timeCreated)
        crc32c = blr.root_object.crc32c or _NA
        md5 = blr.root_object.md5Hash or _NA
    else:
        raise CommandException('Got unexpected URL type (%s)' % url.scheme)
    attrs = [_EncodeUrl(url.url_string), size, time_created, atime, mtime, mode, uid, gid, crc32c, md5]
    attrs = [six.ensure_text(str(i)) for i in attrs]
    return ' '.join(attrs) + '\n'
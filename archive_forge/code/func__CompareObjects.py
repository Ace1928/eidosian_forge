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
def _CompareObjects(self, src_url_str, src_size, src_mtime, src_crc32c, src_md5, dst_url_str, dst_size, dst_mtime, dst_crc32c, dst_md5):
    """Returns whether src should replace dst object, and if mtime is present.

    Uses mtime, size, or whatever checksums are available.

    Args:
      src_url_str: Source URL string.
      src_size: Source size.
      src_mtime: Source modification time.
      src_crc32c: Source CRC32c.
      src_md5: Source MD5.
      dst_url_str: Destination URL string.
      dst_size: Destination size.
      dst_mtime: Destination modification time.
      dst_crc32c: Destination CRC32c.
      dst_md5: Destination MD5.

    Returns:
      A 3-tuple indicating if src should replace dst, and if src and dst have
      mtime.
    """
    has_src_mtime = src_mtime > NA_TIME
    has_dst_mtime = dst_mtime > NA_TIME
    use_hashes = self.compute_file_checksums or (StorageUrlFromString(src_url_str).IsCloudUrl() and StorageUrlFromString(dst_url_str).IsCloudUrl())
    if self.ignore_existing:
        return (False, has_src_mtime, has_dst_mtime)
    if self.skip_old_files and has_src_mtime and has_dst_mtime and (src_mtime < dst_mtime):
        return (False, has_src_mtime, has_dst_mtime)
    if not use_hashes and has_src_mtime and has_dst_mtime:
        return (src_mtime != dst_mtime or src_size != dst_size, has_src_mtime, has_dst_mtime)
    if src_size != dst_size:
        return (True, has_src_mtime, has_dst_mtime)
    src_crc32c, src_md5, dst_crc32c, dst_md5 = _ComputeNeededFileChecksums(self.logger, src_url_str, src_size, src_crc32c, src_md5, dst_url_str, dst_size, dst_crc32c, dst_md5)
    if src_md5 != _NA and dst_md5 != _NA:
        self.logger.debug('Comparing md5 for %s and %s', src_url_str, dst_url_str)
        return (src_md5 != dst_md5, has_src_mtime, has_dst_mtime)
    if src_crc32c != _NA and dst_crc32c != _NA:
        self.logger.debug('Comparing crc32c for %s and %s', src_url_str, dst_url_str)
        return (src_crc32c != dst_crc32c, has_src_mtime, has_dst_mtime)
    if not self._WarnIfMissingCloudHash(src_url_str, src_crc32c, src_md5):
        self._WarnIfMissingCloudHash(dst_url_str, dst_crc32c, dst_md5)
    return (False, has_src_mtime, has_dst_mtime)
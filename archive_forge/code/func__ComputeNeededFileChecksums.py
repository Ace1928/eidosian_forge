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
def _ComputeNeededFileChecksums(logger, src_url_str, src_size, src_crc32c, src_md5, dst_url_str, dst_size, dst_crc32c, dst_md5):
    """Computes any file checksums needed by _CompareObjects.

  Args:
    logger: logging.logger for outputting log messages.
    src_url_str: Source URL string.
    src_size: Source size
    src_crc32c: Source CRC32c.
    src_md5: Source MD5.
    dst_url_str: Destination URL string.
    dst_size: Destination size
    dst_crc32c: Destination CRC32c.
    dst_md5: Destination MD5.

  Returns:
    (src_crc32c, src_md5, dst_crc32c, dst_md5)
  """
    src_url = StorageUrlFromString(src_url_str)
    dst_url = StorageUrlFromString(dst_url_str)
    if src_url.IsFileUrl():
        if dst_crc32c != _NA or dst_url.IsFileUrl():
            if src_size > TEN_MIB:
                logger.info('Computing CRC32C for %s...', src_url_str)
            with open(src_url.object_name, 'rb') as fp:
                src_crc32c = CalculateB64EncodedCrc32cFromContents(fp)
        elif dst_md5 != _NA or dst_url.IsFileUrl():
            if dst_size > TEN_MIB:
                logger.info('Computing MD5 for %s...', src_url_str)
            with open(src_url.object_name, 'rb') as fp:
                src_md5 = CalculateB64EncodedMd5FromContents(fp)
    if dst_url.IsFileUrl():
        if src_crc32c != _NA:
            if src_size > TEN_MIB:
                logger.info('Computing CRC32C for %s...', dst_url_str)
            with open(dst_url.object_name, 'rb') as fp:
                dst_crc32c = CalculateB64EncodedCrc32cFromContents(fp)
        elif src_md5 != _NA:
            if dst_size > TEN_MIB:
                logger.info('Computing MD5 for %s...', dst_url_str)
            with open(dst_url.object_name, 'rb') as fp:
                dst_md5 = CalculateB64EncodedMd5FromContents(fp)
    return (src_crc32c, src_md5, dst_crc32c, dst_md5)
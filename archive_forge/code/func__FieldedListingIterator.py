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
def _FieldedListingIterator(cls, gsutil_api, base_url_str, desc):
    """Iterator over base_url_str formatting output per _BuildTmpOutputLine.

  Args:
    cls: Command instance.
    gsutil_api: gsutil Cloud API instance to use for bucket listing.
    base_url_str: The top-level URL string over which to iterate.
    desc: 'source' or 'destination'.

  Yields:
    Output line formatted per _BuildTmpOutputLine.
  """
    base_url = StorageUrlFromString(base_url_str)
    if base_url.scheme == 'file' and (not cls.recursion_requested):
        iterator = _LocalDirIterator(base_url)
    else:
        if cls.recursion_requested:
            wildcard = '%s/**' % base_url_str.rstrip('/\\')
        else:
            wildcard = '%s/*' % base_url_str.rstrip('/\\')
        fields = ['crc32c', 'md5Hash', 'name', 'size', 'timeCreated', 'metadata/%s' % MTIME_ATTR]
        if cls.preserve_posix_attrs:
            fields.extend(['metadata/%s' % ATIME_ATTR, 'metadata/%s' % MODE_ATTR, 'metadata/%s' % GID_ATTR, 'metadata/%s' % UID_ATTR])
        exclude_tuple = (base_url, cls.exclude_dirs, cls.exclude_pattern) if cls.exclude_pattern is not None else None
        iterator = CreateWildcardIterator(wildcard, gsutil_api, project_id=cls.project_id, exclude_tuple=exclude_tuple, ignore_symlinks=cls.exclude_symlinks, logger=cls.logger).IterObjects(bucket_listing_fields=fields)
    i = 0
    for blr in iterator:
        url = blr.storage_url
        if IsCloudSubdirPlaceholder(url, blr=blr):
            continue
        if cls.exclude_symlinks and url.IsFileUrl() and os.path.islink(url.object_name):
            continue
        if cls.exclude_pattern:
            str_to_check = url.url_string[len(base_url.url_string):]
            if str_to_check.startswith(url.delim):
                str_to_check = str_to_check[1:]
            if cls.exclude_pattern.match(str_to_check):
                continue
        i += 1
        if i % _PROGRESS_REPORT_LISTING_COUNT == 0:
            cls.logger.info('At %s listing %d...', desc, i)
        yield _BuildTmpOutputLine(blr)
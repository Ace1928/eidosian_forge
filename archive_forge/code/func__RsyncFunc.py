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
def _RsyncFunc(cls, diff_to_apply, thread_state=None):
    """Worker function for performing the actual copy and remove operations."""
    gsutil_api = GetCloudApiInstance(cls, thread_state=thread_state)
    dst_url_str = diff_to_apply.dst_url_str
    dst_url = StorageUrlFromString(dst_url_str)
    posix_attrs = diff_to_apply.src_posix_attrs
    if diff_to_apply.diff_action == DiffAction.REMOVE:
        if cls.dryrun:
            cls.logger.info('Would remove %s', dst_url)
        else:
            cls.logger.info('Removing %s', dst_url)
            if dst_url.IsFileUrl():
                try:
                    os.unlink(dst_url.object_name)
                except FileNotFoundError:
                    cls.logger.debug('%s was already removed', dst_url)
                    pass
            else:
                try:
                    gsutil_api.DeleteObject(dst_url.bucket_name, dst_url.object_name, generation=dst_url.generation, provider=dst_url.scheme)
                except NotFoundException:
                    pass
    elif diff_to_apply.diff_action == DiffAction.COPY:
        src_url_str = diff_to_apply.src_url_str
        src_url = StorageUrlFromString(src_url_str)
        if cls.dryrun:
            if src_url.IsFileUrl():
                try:
                    with open(src_url.object_name, 'rb') as _:
                        pass
                except Exception as e:
                    cls.logger.info('Could not open %s' % src_url.object_name)
                    raise
            cls.logger.info('Would copy %s to %s', src_url, dst_url)
        else:
            try:
                src_obj_metadata = None
                if src_url.IsCloudUrl():
                    src_generation = GenerationFromUrlAndString(src_url, src_url.generation)
                    src_obj_metadata = gsutil_api.GetObjectMetadata(src_url.bucket_name, src_url.object_name, generation=src_generation, provider=src_url.scheme, fields=cls.source_metadata_fields)
                    if ObjectIsGzipEncoded(src_obj_metadata):
                        cls.logger.info('%s has a compressed content-encoding, so it will be decompressed upon download; future executions of gsutil rsync with this source object will always download it. If you wish to synchronize such an object efficiently, compress the source objects in place before synchronizing them, rather than (for example) using gsutil cp -Z to compress them on-the-fly (which results in compressed content-encoding).' % src_url)
                else:
                    src_obj_metadata = apitools_messages.Object()
                    if posix_attrs.mtime > long(time.time()) + SECONDS_PER_DAY:
                        WarnFutureTimestamp('mtime', src_url.url_string)
                    if src_url.IsFifo() or src_url.IsStream():
                        type_text = 'Streams' if src_url.IsStream() else 'Named pipes'
                        cls.logger.warn('WARNING: %s are not supported by gsutil rsync and will likely fail. Use the -x option to exclude %s by name.', type_text, src_url.url_string)
                if src_obj_metadata.metadata:
                    custom_metadata = src_obj_metadata.metadata
                else:
                    custom_metadata = apitools_messages.Object.MetadataValue(additionalProperties=[])
                SerializeFileAttributesToObjectMetadata(posix_attrs, custom_metadata, preserve_posix=cls.preserve_posix_attrs)
                tmp_obj_metadata = apitools_messages.Object()
                tmp_obj_metadata.metadata = custom_metadata
                CopyCustomMetadata(tmp_obj_metadata, src_obj_metadata, override=True)
                copy_result = copy_helper.PerformCopy(cls.logger, src_url, dst_url, gsutil_api, cls, _RsyncExceptionHandler, src_obj_metadata=src_obj_metadata, headers=cls.headers, is_rsync=True, gzip_encoded=cls.gzip_encoded, gzip_exts=cls.gzip_exts, preserve_posix=cls.preserve_posix_attrs)
                if copy_result is not None:
                    _, bytes_transferred, _, _ = copy_result
                    with cls.stats_lock:
                        cls.total_bytes_transferred += bytes_transferred
            except SkipUnsupportedObjectError as e:
                cls.logger.info('Skipping item %s with unsupported object type %s', src_url, e.unsupported_type)
    elif diff_to_apply.diff_action == DiffAction.MTIME_SRC_TO_DST:
        dst_url = StorageUrlFromString(diff_to_apply.dst_url_str)
        if cls.dryrun:
            cls.logger.info('Would set mtime for %s', dst_url)
        else:
            cls.logger.info('Copying mtime from src to dst for %s', dst_url.url_string)
            mtime = posix_attrs.mtime
            obj_metadata = apitools_messages.Object()
            obj_metadata.metadata = CreateCustomMetadata({MTIME_ATTR: mtime})
            if dst_url.IsCloudUrl():
                dst_url = StorageUrlFromString(diff_to_apply.dst_url_str)
                dst_generation = GenerationFromUrlAndString(dst_url, dst_url.generation)
                try:
                    gsutil_api.PatchObjectMetadata(dst_url.bucket_name, dst_url.object_name, obj_metadata, provider=dst_url.scheme, generation=dst_url.generation)
                except ServiceException as err:
                    cls.logger.debug('Error while trying to patch: %s', err)
                    cls.logger.info("Copying whole file/object for %s instead of patching because you don't have patch permission on the object.", dst_url.url_string)
                    _RsyncFunc(cls, RsyncDiffToApply(diff_to_apply.src_url_str, diff_to_apply.dst_url_str, posix_attrs, DiffAction.COPY, diff_to_apply.copy_size), thread_state=thread_state)
            else:
                ParseAndSetPOSIXAttributes(dst_url.object_name, obj_metadata, preserve_posix=cls.preserve_posix_attrs)
    elif diff_to_apply.diff_action == DiffAction.POSIX_SRC_TO_DST:
        dst_url = StorageUrlFromString(diff_to_apply.dst_url_str)
        if cls.dryrun:
            cls.logger.info('Would set POSIX attributes for %s', dst_url)
        else:
            cls.logger.info('Copying POSIX attributes from src to dst for %s', dst_url.url_string)
            obj_metadata = apitools_messages.Object()
            obj_metadata.metadata = apitools_messages.Object.MetadataValue(additionalProperties=[])
            SerializeFileAttributesToObjectMetadata(posix_attrs, obj_metadata.metadata, preserve_posix=True)
            if dst_url.IsCloudUrl():
                dst_generation = GenerationFromUrlAndString(dst_url, dst_url.generation)
                dst_obj_metadata = gsutil_api.GetObjectMetadata(dst_url.bucket_name, dst_url.object_name, generation=dst_generation, provider=dst_url.scheme, fields=['acl'])
                try:
                    gsutil_api.PatchObjectMetadata(dst_url.bucket_name, dst_url.object_name, obj_metadata, provider=dst_url.scheme, generation=dst_url.generation)
                except ServiceException as err:
                    cls.logger.debug('Error while trying to patch: %s', err)
                    cls.logger.info("Copying whole file/object for %s instead of patching because you don't have patch permission on the object.", dst_url.url_string)
                    _RsyncFunc(cls, RsyncDiffToApply(diff_to_apply.src_url_str, diff_to_apply.dst_url_str, posix_attrs, DiffAction.COPY, diff_to_apply.copy_size), thread_state=thread_state)
    else:
        raise CommandException('Got unexpected DiffAction (%d)' % diff_to_apply.diff_action)
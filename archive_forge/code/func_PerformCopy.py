from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import base64
from collections import namedtuple
import csv
import datetime
import errno
import gzip
import json
import logging
import mimetypes
from operator import attrgetter
import os
import pickle
import pyu2f
import random
import re
import shutil
import six
import stat
import subprocess
import tempfile
import textwrap
import time
import traceback
import six
from six.moves import xrange
from six.moves import range
from apitools.base.protorpclite import protojson
from boto import config
import crcmod
import gslib
from gslib.cloud_api import AccessDeniedException
from gslib.cloud_api import ArgumentException
from gslib.cloud_api import CloudApi
from gslib.cloud_api import EncryptionException
from gslib.cloud_api import NotFoundException
from gslib.cloud_api import PreconditionException
from gslib.cloud_api import Preconditions
from gslib.cloud_api import ResumableDownloadException
from gslib.cloud_api import ResumableUploadAbortException
from gslib.cloud_api import ResumableUploadException
from gslib.cloud_api import ResumableUploadStartOverException
from gslib.cloud_api import ServiceException
from gslib.commands.compose import MAX_COMPOSE_ARITY
from gslib.commands.config import DEFAULT_PARALLEL_COMPOSITE_UPLOAD_COMPONENT_SIZE
from gslib.commands.config import DEFAULT_PARALLEL_COMPOSITE_UPLOAD_THRESHOLD
from gslib.commands.config import DEFAULT_SLICED_OBJECT_DOWNLOAD_COMPONENT_SIZE
from gslib.commands.config import DEFAULT_SLICED_OBJECT_DOWNLOAD_MAX_COMPONENTS
from gslib.commands.config import DEFAULT_SLICED_OBJECT_DOWNLOAD_THRESHOLD
from gslib.commands.config import DEFAULT_GZIP_COMPRESSION_LEVEL
from gslib.cs_api_map import ApiSelector
from gslib.daisy_chain_wrapper import DaisyChainWrapper
from gslib.exception import CommandException
from gslib.exception import HashMismatchException
from gslib.exception import InvalidUrlError
from gslib.file_part import FilePart
from gslib.parallel_tracker_file import GenerateComponentObjectPrefix
from gslib.parallel_tracker_file import ReadParallelUploadTrackerFile
from gslib.parallel_tracker_file import ValidateParallelCompositeTrackerData
from gslib.parallel_tracker_file import WriteComponentToParallelUploadTrackerFile
from gslib.parallel_tracker_file import WriteParallelUploadTrackerFile
from gslib.progress_callback import FileProgressCallbackHandler
from gslib.progress_callback import ProgressCallbackWithTimeout
from gslib.resumable_streaming_upload import ResumableStreamingJsonUploadWrapper
from gslib import storage_url
from gslib.storage_url import ContainsWildcard
from gslib.storage_url import GenerationFromUrlAndString
from gslib.storage_url import IsCloudSubdirPlaceholder
from gslib.storage_url import StorageUrlFromString
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.thread_message import FileMessage
from gslib.thread_message import RetryableErrorMessage
from gslib.tracker_file import DeleteDownloadTrackerFiles
from gslib.tracker_file import DeleteTrackerFile
from gslib.tracker_file import ENCRYPTION_UPLOAD_TRACKER_ENTRY
from gslib.tracker_file import GetDownloadStartByte
from gslib.tracker_file import GetTrackerFilePath
from gslib.tracker_file import GetUploadTrackerData
from gslib.tracker_file import RaiseUnwritableTrackerFileException
from gslib.tracker_file import ReadOrCreateDownloadTrackerFile
from gslib.tracker_file import SERIALIZATION_UPLOAD_TRACKER_ENTRY
from gslib.tracker_file import TrackerFileType
from gslib.tracker_file import WriteDownloadComponentTrackerFile
from gslib.tracker_file import WriteJsonDataToTrackerFile
from gslib.utils import parallelism_framework_util
from gslib.utils import stet_util
from gslib.utils import temporary_file_util
from gslib.utils import text_util
from gslib.utils.boto_util import GetJsonResumableChunkSize
from gslib.utils.boto_util import GetMaxRetryDelay
from gslib.utils.boto_util import GetNumRetries
from gslib.utils.boto_util import ResumableThreshold
from gslib.utils.boto_util import UsingCrcmodExtension
from gslib.utils.cloud_api_helper import GetCloudApiInstance
from gslib.utils.cloud_api_helper import GetDownloadSerializationData
from gslib.utils.constants import DEFAULT_FILE_BUFFER_SIZE
from gslib.utils.constants import MIN_SIZE_COMPUTE_LOGGING
from gslib.utils.constants import UTF8
from gslib.utils.encryption_helper import CryptoKeyType
from gslib.utils.encryption_helper import CryptoKeyWrapperFromKey
from gslib.utils.encryption_helper import FindMatchingCSEKInBotoConfig
from gslib.utils.encryption_helper import GetEncryptionKeyWrapper
from gslib.utils.hashing_helper import Base64EncodeHash
from gslib.utils.hashing_helper import CalculateB64EncodedMd5FromContents
from gslib.utils.hashing_helper import CalculateHashesFromContents
from gslib.utils.hashing_helper import CHECK_HASH_IF_FAST_ELSE_FAIL
from gslib.utils.hashing_helper import CHECK_HASH_NEVER
from gslib.utils.hashing_helper import ConcatCrc32c
from gslib.utils.hashing_helper import GetDownloadHashAlgs
from gslib.utils.hashing_helper import GetMd5
from gslib.utils.hashing_helper import GetUploadHashAlgs
from gslib.utils.hashing_helper import HashingFileUploadWrapper
from gslib.utils.metadata_util import ObjectIsGzipEncoded
from gslib.utils.parallelism_framework_util import AtomicDict
from gslib.utils.parallelism_framework_util import CheckMultiprocessingAvailableAndInit
from gslib.utils.parallelism_framework_util import PutToQueueWithTimeout
from gslib.utils.posix_util import ATIME_ATTR
from gslib.utils.posix_util import ConvertDatetimeToPOSIX
from gslib.utils.posix_util import GID_ATTR
from gslib.utils.posix_util import MODE_ATTR
from gslib.utils.posix_util import MTIME_ATTR
from gslib.utils.posix_util import ParseAndSetPOSIXAttributes
from gslib.utils.posix_util import UID_ATTR
from gslib.utils.system_util import CheckFreeSpace
from gslib.utils.system_util import GetFileSize
from gslib.utils.system_util import GetStreamFromFileUrl
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils.translation_helper import AddS3MarkerAclToObjectMetadata
from gslib.utils.translation_helper import CopyObjectMetadata
from gslib.utils.translation_helper import DEFAULT_CONTENT_TYPE
from gslib.utils.translation_helper import ObjectMetadataFromHeaders
from gslib.utils.translation_helper import PreconditionsFromHeaders
from gslib.utils.translation_helper import S3MarkerAclFromObjectMetadata
from gslib.utils.unit_util import DivideAndCeil
from gslib.utils.unit_util import HumanReadableToBytes
from gslib.utils.unit_util import MakeHumanReadable
from gslib.utils.unit_util import SECONDS_PER_DAY
from gslib.utils.unit_util import TEN_MIB
from gslib.wildcard_iterator import CreateWildcardIterator
def PerformCopy(logger, src_url, dst_url, gsutil_api, command_obj, copy_exception_handler, src_obj_metadata=None, allow_splitting=True, headers=None, manifest=None, gzip_exts=None, is_rsync=False, preserve_posix=False, gzip_encoded=False, use_stet=False):
    """Performs copy from src_url to dst_url, handling various special cases.

  Args:
    logger: for outputting log messages.
    src_url: Source StorageUrl.
    dst_url: Destination StorageUrl.
    gsutil_api: gsutil Cloud API instance to use for the copy.
    command_obj: command object for use in Apply in parallel composite uploads
        and sliced object downloads.
    copy_exception_handler: for handling copy exceptions during Apply.
    src_obj_metadata: If source URL is a cloud object, source object metadata
        with all necessary fields (per GetSourceFieldsNeededForCopy).
        Required for cloud source URLs. If source URL is a file, an
        apitools Object that may contain file size, or None.
    allow_splitting: Whether to allow the file to be split into component
                     pieces for an parallel composite upload or download.
    headers: optional headers to use for the copy operation.
    manifest: optional manifest for tracking copy operations.
    gzip_exts: List of file extensions to gzip, if any.
               If gzip_exts is GZIP_ALL_FILES, gzip all files.
    is_rsync: Whether or not the caller is the rsync command.
    preserve_posix: Whether or not to preserve posix attributes.
    gzip_encoded: Whether to use gzip transport encoding for the upload. Used
        in conjunction with gzip_exts. Streaming files compressed is only
        supported on the JSON GCS API.
    use_stet: If True, will perform STET encryption or decryption using
        the binary specified in the boto config or PATH.

  Returns:
    (elapsed_time, bytes_transferred, version-specific dst_url) excluding
    overhead like initial GET.

  Raises:
    ItemExistsError: if no clobber flag is specified and the destination
        object already exists.
    SkipUnsupportedObjectError: if skip_unsupported_objects flag is specified
        and the source is an unsupported type.
    CommandException: if other errors encountered.
  """
    if headers:
        dst_obj_headers = headers.copy()
    else:
        dst_obj_headers = {}
    dst_obj_metadata = ObjectMetadataFromHeaders(dst_obj_headers)
    if dst_url.IsCloudUrl() and dst_url.scheme == 'gs':
        preconditions = PreconditionsFromHeaders(dst_obj_headers)
    else:
        preconditions = Preconditions()
    src_obj_filestream = None
    decryption_key = None
    copy_in_the_cloud = False
    if src_url.IsCloudUrl():
        if dst_url.IsCloudUrl() and src_url.scheme == dst_url.scheme and (not global_copy_helper_opts.daisy_chain):
            copy_in_the_cloud = True
        if global_copy_helper_opts.perform_mv:
            WarnIfMvEarlyDeletionChargeApplies(src_url, src_obj_metadata, logger)
        MaybeSkipUnsupportedObject(src_url, src_obj_metadata)
        decryption_key = GetDecryptionCSEK(src_url, src_obj_metadata)
        src_obj_size = src_obj_metadata.size
        dst_obj_metadata.contentType = src_obj_metadata.contentType
        if global_copy_helper_opts.preserve_acl and dst_url.IsCloudUrl():
            if src_url.scheme == 'gs' and (not src_obj_metadata.acl):
                raise CommandException('No OWNER permission found for object %s. OWNER permission is required for preserving ACLs.' % src_url)
            dst_obj_metadata.acl = src_obj_metadata.acl
            if src_url.scheme == 's3':
                acl_text = S3MarkerAclFromObjectMetadata(src_obj_metadata)
                if acl_text:
                    AddS3MarkerAclToObjectMetadata(dst_obj_metadata, acl_text)
    else:
        if use_stet:
            source_stream_url = stet_util.encrypt_upload(src_url, dst_url, logger)
        else:
            source_stream_url = src_url
        try:
            src_obj_filestream = GetStreamFromFileUrl(source_stream_url)
        except Exception as e:
            message = 'Error opening file "%s": %s.' % (src_url, str(e))
            if command_obj.continue_on_error:
                command_obj.op_failure_count += 1
                logger.error(message)
                return
            else:
                raise CommandException(message)
        if src_url.IsStream() or src_url.IsFifo():
            src_obj_size = None
        elif src_obj_metadata and src_obj_metadata.size and (not use_stet):
            src_obj_size = src_obj_metadata.size
        else:
            src_obj_size = os.path.getsize(source_stream_url.object_name)
    if global_copy_helper_opts.use_manifest:
        manifest.Set(src_url.url_string, 'size', src_obj_size)
    if dst_url.scheme == 's3' and src_url != 's3' and (src_obj_size is not None) and (src_obj_size > S3_MAX_UPLOAD_SIZE):
        raise CommandException('"%s" exceeds the maximum gsutil-supported size for an S3 upload. S3 objects greater than %s in size require multipart uploads, which gsutil does not support.' % (src_url, MakeHumanReadable(S3_MAX_UPLOAD_SIZE)))
    if IS_WINDOWS and src_url.IsFileUrl() and src_url.IsStream():
        msvcrt.setmode(GetStreamFromFileUrl(src_url).fileno(), os.O_BINARY)
    if global_copy_helper_opts.no_clobber:
        if preconditions.gen_match:
            raise ArgumentException('Specifying x-goog-if-generation-match is not supported with cp -n')
        else:
            preconditions.gen_match = 0
        if dst_url.IsFileUrl() and os.path.exists(dst_url.object_name):
            raise ItemExistsError()
        elif dst_url.IsCloudUrl():
            try:
                dst_object = gsutil_api.GetObjectMetadata(dst_url.bucket_name, dst_url.object_name, provider=dst_url.scheme)
            except NotFoundException:
                dst_object = None
            if dst_object:
                raise ItemExistsError()
    if dst_url.IsCloudUrl():
        dst_obj_metadata.name = dst_url.object_name
        dst_obj_metadata.bucket = dst_url.bucket_name
        if src_url.IsCloudUrl():
            src_obj_metadata.name = src_url.object_name
            src_obj_metadata.bucket = src_url.bucket_name
        else:
            _SetContentTypeFromFile(src_url, dst_obj_metadata)
        encryption_keywrapper = GetEncryptionKeyWrapper(config)
        if encryption_keywrapper and encryption_keywrapper.crypto_type == CryptoKeyType.CMEK and (dst_url.scheme == 'gs'):
            dst_obj_metadata.kmsKeyName = encryption_keywrapper.crypto_key
    if src_obj_metadata:
        CopyObjectMetadata(src_obj_metadata, dst_obj_metadata, override=False)
    if global_copy_helper_opts.dest_storage_class:
        dst_obj_metadata.storageClass = global_copy_helper_opts.dest_storage_class
    if config.get('GSUtil', 'check_hashes') == CHECK_HASH_NEVER:
        dst_obj_metadata.md5Hash = None
    _LogCopyOperation(logger, src_url, dst_url, dst_obj_metadata)
    if src_url.IsCloudUrl():
        if dst_url.IsFileUrl():
            PutToQueueWithTimeout(gsutil_api.status_queue, FileMessage(src_url, dst_url, time.time(), message_type=FileMessage.FILE_DOWNLOAD, size=src_obj_size, finished=False))
            return _DownloadObjectToFile(src_url, src_obj_metadata, dst_url, gsutil_api, logger, command_obj, copy_exception_handler, allow_splitting=allow_splitting, decryption_key=decryption_key, is_rsync=is_rsync, preserve_posix=preserve_posix, use_stet=use_stet)
        elif copy_in_the_cloud:
            PutToQueueWithTimeout(gsutil_api.status_queue, FileMessage(src_url, dst_url, time.time(), message_type=FileMessage.FILE_CLOUD_COPY, size=src_obj_size, finished=False))
            return _CopyObjToObjInTheCloud(src_url, src_obj_metadata, dst_url, dst_obj_metadata, preconditions, gsutil_api, decryption_key=decryption_key)
        else:
            PutToQueueWithTimeout(gsutil_api.status_queue, FileMessage(src_url, dst_url, time.time(), message_type=FileMessage.FILE_DAISY_COPY, size=src_obj_size, finished=False))
            return _CopyObjToObjDaisyChainMode(src_url, src_obj_metadata, dst_url, dst_obj_metadata, preconditions, gsutil_api, logger, decryption_key=decryption_key)
    elif dst_url.IsCloudUrl():
        uploaded_metadata = _UploadFileToObject(src_url, src_obj_filestream, src_obj_size, dst_url, dst_obj_metadata, preconditions, gsutil_api, logger, command_obj, copy_exception_handler, gzip_exts=gzip_exts, allow_splitting=allow_splitting, gzip_encoded=gzip_encoded)
        if use_stet:
            os.unlink(src_obj_filestream.name)
        return uploaded_metadata
    else:
        PutToQueueWithTimeout(gsutil_api.status_queue, FileMessage(src_url, dst_url, time.time(), message_type=FileMessage.FILE_LOCAL_COPY, size=src_obj_size, finished=False))
        result = _CopyFileToFile(src_url, dst_url, status_queue=gsutil_api.status_queue, src_obj_metadata=src_obj_metadata)
        if not src_url.IsStream() and (not dst_url.IsStream()):
            ParseAndSetPOSIXAttributes(dst_url.object_name, src_obj_metadata, is_rsync=is_rsync, preserve_posix=preserve_posix)
        return result
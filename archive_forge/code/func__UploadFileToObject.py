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
def _UploadFileToObject(src_url, src_obj_filestream, src_obj_size, dst_url, dst_obj_metadata, preconditions, gsutil_api, logger, command_obj, copy_exception_handler, gzip_exts=None, allow_splitting=True, is_component=False, gzip_encoded=False):
    """Uploads a local file to an object.

  Args:
    src_url: Source FileUrl.
    src_obj_filestream: Read stream of the source file to be read and closed.
    src_obj_size (int or None): Size of the source file.
    dst_url: Destination CloudUrl.
    dst_obj_metadata: Metadata to be applied to the destination object.
    preconditions: Preconditions to use for the copy.
    gsutil_api: gsutil Cloud API to use for the copy.
    logger: for outputting log messages.
    command_obj: command object for use in Apply in parallel composite uploads.
    copy_exception_handler: For handling copy exceptions during Apply.
    gzip_exts: List of file extensions to gzip prior to upload, if any.
               If gzip_exts is GZIP_ALL_FILES, gzip all files.
    allow_splitting: Whether to allow the file to be split into component
                     pieces for an parallel composite upload.
    is_component: indicates whether this is a single component or whole file.
    gzip_encoded: Whether to use gzip transport encoding for the upload. Used
        in conjunction with gzip_exts for selecting which files will be
        encoded. Streaming files compressed is only supported on the JSON GCS
        API.

  Returns:
    (elapsed_time, bytes_transferred, dst_url with generation,
    md5 hash of destination) excluding overhead like initial GET.

  Raises:
    CommandException: if errors encountered.
  """
    if not dst_obj_metadata or not dst_obj_metadata.contentLanguage:
        content_language = config.get_value('GSUtil', 'content_language')
        if content_language:
            dst_obj_metadata.contentLanguage = content_language
    upload_url = src_url
    upload_stream = src_obj_filestream
    upload_size = src_obj_size
    zipped_file, gzip_encoded_file = _SelectUploadCompressionStrategy(src_url.object_name, is_component, gzip_exts, gzip_encoded)
    if gzip_encoded_file and (not is_component):
        logger.debug('Using compressed transport encoding for %s.', src_url)
    elif zipped_file:
        upload_url, upload_stream, upload_size = _ApplyZippedUploadCompression(src_url, src_obj_filestream, src_obj_size, logger)
        dst_obj_metadata.contentEncoding = 'gzip'
        if not dst_obj_metadata.cacheControl:
            dst_obj_metadata.cacheControl = 'no-transform'
        elif 'no-transform' not in dst_obj_metadata.cacheControl.lower():
            dst_obj_metadata.cacheControl += ',no-transform'
    if not is_component:
        PutToQueueWithTimeout(gsutil_api.status_queue, FileMessage(upload_url, dst_url, time.time(), message_type=FileMessage.FILE_UPLOAD, size=upload_size, finished=False))
    elapsed_time = None
    uploaded_object = None
    hash_algs = GetUploadHashAlgs()
    digesters = dict(((alg, hash_algs[alg]()) for alg in hash_algs or {}))
    parallel_composite_upload = _ShouldDoParallelCompositeUpload(logger, allow_splitting, upload_url, dst_url, src_obj_size, gsutil_api, canned_acl=global_copy_helper_opts.canned_acl)
    non_resumable_upload = (0 if upload_size is None else upload_size) < ResumableThreshold() or src_url.IsStream() or src_url.IsFifo()
    if (src_url.IsStream() or src_url.IsFifo()) and gsutil_api.GetApiSelector(provider=dst_url.scheme) == ApiSelector.JSON:
        orig_stream = upload_stream
        upload_stream = ResumableStreamingJsonUploadWrapper(orig_stream, GetJsonResumableChunkSize())
    if not parallel_composite_upload and len(hash_algs):
        wrapped_filestream = HashingFileUploadWrapper(upload_stream, digesters, hash_algs, upload_url, logger)
    else:
        wrapped_filestream = upload_stream

    def CallParallelCompositeUpload():
        return _DoParallelCompositeUpload(upload_stream, upload_url, dst_url, dst_obj_metadata, global_copy_helper_opts.canned_acl, upload_size, preconditions, gsutil_api, command_obj, copy_exception_handler, logger, gzip_encoded=gzip_encoded_file)

    def CallNonResumableUpload():
        return _UploadFileToObjectNonResumable(upload_url, wrapped_filestream, upload_size, dst_url, dst_obj_metadata, preconditions, gsutil_api, gzip_encoded=gzip_encoded_file)

    def CallResumableUpload():
        return _UploadFileToObjectResumable(upload_url, wrapped_filestream, upload_size, dst_url, dst_obj_metadata, preconditions, gsutil_api, logger, is_component=is_component, gzip_encoded=gzip_encoded_file)
    if parallel_composite_upload:
        delegate = CallParallelCompositeUpload
    elif non_resumable_upload:
        delegate = CallNonResumableUpload
    else:
        delegate = CallResumableUpload
    elapsed_time, uploaded_object = _DelegateUploadFileToObject(delegate, upload_url, upload_stream, zipped_file, gzip_encoded_file, parallel_composite_upload, logger)
    if not parallel_composite_upload:
        try:
            digests = _CreateDigestsFromDigesters(digesters)
            _CheckHashes(logger, dst_url, uploaded_object, src_url.object_name, digests, is_upload=True)
        except HashMismatchException:
            if _RENAME_ON_HASH_MISMATCH:
                corrupted_obj_metadata = apitools_messages.Object(name=dst_obj_metadata.name, bucket=dst_obj_metadata.bucket, etag=uploaded_object.etag)
                dst_obj_metadata.name = dst_url.object_name + _RENAME_ON_HASH_MISMATCH_SUFFIX
                gsutil_api.CopyObject(corrupted_obj_metadata, dst_obj_metadata, provider=dst_url.scheme)
            gsutil_api.DeleteObject(dst_url.bucket_name, dst_url.object_name, generation=uploaded_object.generation, provider=dst_url.scheme)
            raise
    result_url = dst_url.Clone()
    result_url.generation = uploaded_object.generation
    result_url.generation = GenerationFromUrlAndString(result_url, uploaded_object.generation)
    if not is_component:
        PutToQueueWithTimeout(gsutil_api.status_queue, FileMessage(upload_url, dst_url, time.time(), message_type=FileMessage.FILE_UPLOAD, size=upload_size, finished=True))
    return (elapsed_time, uploaded_object.size, result_url, uploaded_object.md5Hash)
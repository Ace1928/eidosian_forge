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
def _DownloadObjectToFile(src_url, src_obj_metadata, dst_url, gsutil_api, logger, command_obj, copy_exception_handler, allow_splitting=True, decryption_key=None, is_rsync=False, preserve_posix=False, use_stet=False):
    """Downloads an object to a local file.

  Args:
    src_url: Source CloudUrl.
    src_obj_metadata: Metadata from the source object.
    dst_url: Destination FileUrl.
    gsutil_api: gsutil Cloud API instance to use for the download.
    logger: for outputting log messages.
    command_obj: command object for use in Apply in sliced downloads.
    copy_exception_handler: For handling copy exceptions during Apply.
    allow_splitting: Whether or not to allow sliced download.
    decryption_key: Base64-encoded decryption key for the source object, if any.
    is_rsync: Whether or not the caller is the rsync command.
    preserve_posix: Whether or not to preserve POSIX attributes.
    use_stet: Decrypt downloaded file with STET binary if available on system.

  Returns:
    (elapsed_time, bytes_transferred, dst_url, md5), where time elapsed
    excludes initial GET.

  Raises:
    FileConcurrencySkipError: if this download is already in progress.
    CommandException: if other errors encountered.
  """
    global open_files_map, open_files_lock
    if dst_url.object_name.endswith(dst_url.delim):
        logger.warn('\n'.join(textwrap.wrap('Skipping attempt to download to filename ending with slash (%s). This typically happens when using gsutil to download from a subdirectory created by the Cloud Console (https://cloud.google.com/console)' % dst_url.object_name)))
        raise InvalidUrlError('Invalid destination path: %s' % dst_url.object_name)
    api_selector = gsutil_api.GetApiSelector(provider=src_url.scheme)
    download_strategy = _SelectDownloadStrategy(dst_url)
    sliced_download = _ShouldDoSlicedDownload(download_strategy, src_obj_metadata, allow_splitting, logger)
    download_file_name, need_to_unzip = _GetDownloadFile(dst_url, src_obj_metadata, logger)
    with open_files_lock:
        if open_files_map.get(download_file_name, False):
            raise FileConcurrencySkipError
        open_files_map[download_file_name] = True
    consider_md5 = src_obj_metadata.md5Hash and (not sliced_download)
    hash_algs = GetDownloadHashAlgs(logger, consider_md5=consider_md5, consider_crc32c=src_obj_metadata.crc32c)
    digesters = dict(((alg, hash_algs[alg]()) for alg in hash_algs or {}))
    server_encoding = None
    download_complete = src_obj_metadata.size == 0
    bytes_transferred = 0
    start_time = time.time()
    if not download_complete:
        if sliced_download:
            bytes_transferred, crc32c = _DoSlicedDownload(src_url, src_obj_metadata, dst_url, download_file_name, command_obj, logger, copy_exception_handler, api_selector, decryption_key=decryption_key, status_queue=gsutil_api.status_queue)
            if 'crc32c' in digesters:
                digesters['crc32c'].crcValue = crc32c
        elif download_strategy is CloudApi.DownloadStrategy.ONE_SHOT:
            bytes_transferred, server_encoding = _DownloadObjectToFileNonResumable(src_url, src_obj_metadata, dst_url, download_file_name, gsutil_api, digesters, decryption_key=decryption_key)
        elif download_strategy is CloudApi.DownloadStrategy.RESUMABLE:
            bytes_transferred, server_encoding = _DownloadObjectToFileResumable(src_url, src_obj_metadata, dst_url, download_file_name, gsutil_api, logger, digesters, decryption_key=decryption_key)
        else:
            raise CommandException('Invalid download strategy %s chosen forfile %s' % (download_strategy, download_file_name))
    end_time = time.time()
    server_gzip = server_encoding and server_encoding.lower().endswith('gzip')
    local_md5 = _ValidateAndCompleteDownload(logger, src_url, src_obj_metadata, dst_url, need_to_unzip, server_gzip, digesters, hash_algs, download_file_name, api_selector, bytes_transferred, gsutil_api, is_rsync=is_rsync, preserve_posix=preserve_posix, use_stet=use_stet)
    with open_files_lock:
        open_files_map.delete(download_file_name)
    PutToQueueWithTimeout(gsutil_api.status_queue, FileMessage(src_url, dst_url, message_time=end_time, message_type=FileMessage.FILE_DOWNLOAD, size=src_obj_metadata.size, finished=True))
    return (end_time - start_time, bytes_transferred, dst_url, local_md5)
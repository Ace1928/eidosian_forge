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
def _DownloadObjectToFileResumable(src_url, src_obj_metadata, dst_url, download_file_name, gsutil_api, logger, digesters, component_num=None, start_byte=0, end_byte=None, decryption_key=None):
    """Downloads an object to a local file using the resumable strategy.

  Args:
    src_url: Source CloudUrl.
    src_obj_metadata: Metadata from the source object.
    dst_url: Destination FileUrl.
    download_file_name: Temporary file name to be used for download.
    gsutil_api: gsutil Cloud API instance to use for the download.
    logger: for outputting log messages.
    digesters: Digesters corresponding to the hash algorithms that will be used
               for validation.
    component_num: Which component of a sliced download this call is for, or
                   None if this is not a sliced download.
    start_byte: The first byte of a byte range for a sliced download.
    end_byte: The last byte of a byte range for a sliced download.
    decryption_key: Base64-encoded decryption key for the source object, if any.

  Returns:
    (bytes_transferred, server_encoding)
    bytes_transferred: Number of bytes transferred from server this call.
    server_encoding: Content-encoding string if it was detected that the server
                     sent encoded bytes during transfer, None otherwise.
  """
    if end_byte is None:
        end_byte = src_obj_metadata.size - 1
    download_size = end_byte - start_byte + 1
    is_sliced = component_num is not None
    api_selector = gsutil_api.GetApiSelector(provider=src_url.scheme)
    server_encoding = None
    download_name = dst_url.object_name
    if is_sliced:
        download_name += ' component %d' % component_num
    fp = None
    try:
        fp = open(download_file_name, 'r+b')
        fp.seek(start_byte)
        api_selector = gsutil_api.GetApiSelector(provider=src_url.scheme)
        existing_file_size = GetFileSize(fp)
        tracker_file_name, download_start_byte = ReadOrCreateDownloadTrackerFile(src_obj_metadata, dst_url, logger, api_selector, start_byte, existing_file_size, component_num)
        if download_start_byte < start_byte or download_start_byte > end_byte + 1:
            DeleteTrackerFile(tracker_file_name)
            raise CommandException('Resumable download start point for %s is not in the correct byte range. Deleting tracker file, so if you re-try this download it will start from scratch' % download_name)
        download_complete = download_start_byte == start_byte + download_size
        resuming = download_start_byte != start_byte and (not download_complete)
        if resuming:
            logger.info('Resuming download for %s', download_name)
        elif download_complete:
            logger.info('Download already complete for %s, skipping download but will run integrity checks.', download_name)
        serialization_data = GetDownloadSerializationData(src_obj_metadata, progress=download_start_byte, user_project=gsutil_api.user_project)
        if resuming or download_complete:
            bytes_digested = 0
            total_bytes_to_digest = download_start_byte - start_byte
            hash_callback = ProgressCallbackWithTimeout(total_bytes_to_digest, FileProgressCallbackHandler(gsutil_api.status_queue, component_num=component_num, src_url=src_url, dst_url=dst_url, operation_name='Hashing').call)
            while bytes_digested < total_bytes_to_digest:
                bytes_to_read = min(DEFAULT_FILE_BUFFER_SIZE, total_bytes_to_digest - bytes_digested)
                data = fp.read(bytes_to_read)
                bytes_digested += bytes_to_read
                for alg_name in digesters:
                    digesters[alg_name].update(six.ensure_binary(data))
                hash_callback.Progress(len(data))
        elif not is_sliced:
            fp.truncate(0)
            existing_file_size = 0
        progress_callback = FileProgressCallbackHandler(gsutil_api.status_queue, start_byte=start_byte, override_total_size=download_size, src_url=src_url, dst_url=dst_url, component_num=component_num, operation_name='Downloading').call
        if global_copy_helper_opts.test_callback_file:
            with open(global_copy_helper_opts.test_callback_file, 'rb') as test_fp:
                progress_callback = pickle.loads(test_fp.read()).call
        if is_sliced and src_obj_metadata.size >= ResumableThreshold():
            fp = SlicedDownloadFileWrapper(fp, tracker_file_name, src_obj_metadata, start_byte, end_byte)
        compressed_encoding = ObjectIsGzipEncoded(src_obj_metadata)
        if not download_complete:
            fp.seek(download_start_byte)
            server_encoding = gsutil_api.GetObjectMedia(src_url.bucket_name, src_url.object_name, fp, start_byte=download_start_byte, end_byte=end_byte, compressed_encoding=compressed_encoding, generation=src_url.generation, object_size=src_obj_metadata.size, download_strategy=CloudApi.DownloadStrategy.RESUMABLE, provider=src_url.scheme, serialization_data=serialization_data, digesters=digesters, progress_callback=progress_callback, decryption_tuple=CryptoKeyWrapperFromKey(decryption_key))
    except ResumableDownloadException as e:
        logger.warning('Caught ResumableDownloadException (%s) for download of %s.', e.reason, download_name)
        raise
    finally:
        if fp:
            fp.close()
    bytes_transferred = end_byte - download_start_byte + 1
    return (bytes_transferred, server_encoding)
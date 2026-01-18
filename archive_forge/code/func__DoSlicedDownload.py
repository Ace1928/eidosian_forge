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
def _DoSlicedDownload(src_url, src_obj_metadata, dst_url, download_file_name, command_obj, logger, copy_exception_handler, api_selector, decryption_key=None, status_queue=None):
    """Downloads a cloud object to a local file using sliced download.

  Byte ranges are decided for each thread/process, and then the parts are
  downloaded in parallel.

  Args:
    src_url: Source CloudUrl.
    src_obj_metadata: Metadata from the source object.
    dst_url: Destination FileUrl.
    download_file_name: Temporary file name to be used for download.
    command_obj: command object for use in Apply in parallel composite uploads.
    logger: for outputting log messages.
    copy_exception_handler: For handling copy exceptions during Apply.
    api_selector: The Cloud API implementation used.
    decryption_key: Base64-encoded decryption key for the source object, if any.
    status_queue: Queue for posting file messages for UI/Analytics.

  Returns:
    (bytes_transferred, crc32c)
    bytes_transferred: Number of bytes transferred from server this call.
    crc32c: a crc32c hash value (integer) for the downloaded bytes, or None if
            crc32c hashing wasn't performed.
  """
    src_obj_metadata.customerEncryption = None
    components_to_download, component_lengths = _PartitionObject(src_url, src_obj_metadata, dst_url, download_file_name, decryption_key)
    num_components = len(components_to_download)
    _MaintainSlicedDownloadTrackerFiles(src_obj_metadata, dst_url, download_file_name, logger, api_selector, num_components)
    with open(download_file_name, 'r+b') as fp:
        fp.truncate(src_obj_metadata.size)
    for i, component in enumerate(components_to_download):
        size = component.end_byte - component.start_byte + 1
        download_start_byte = GetDownloadStartByte(src_obj_metadata, dst_url, api_selector, component.start_byte, size, i)
        bytes_already_downloaded = download_start_byte - component.start_byte
        PutToQueueWithTimeout(status_queue, FileMessage(src_url, dst_url, time.time(), size=size, finished=False, component_num=i, message_type=FileMessage.COMPONENT_TO_DOWNLOAD, bytes_already_downloaded=bytes_already_downloaded))
    cp_results = command_obj.Apply(_PerformSlicedDownloadObjectToFile, components_to_download, copy_exception_handler, arg_checker=gslib.command.DummyArgChecker, parallel_operations_override=command_obj.ParallelOverrideReason.SLICE, should_return_results=True)
    if len(cp_results) < num_components:
        raise CommandException('Some components of %s were not downloaded successfully. Please retry this download.' % dst_url.object_name)
    cp_results = sorted(cp_results, key=attrgetter('component_num'))
    crc32c = cp_results[0].crc32c
    if crc32c is not None:
        for i in range(1, num_components):
            crc32c = ConcatCrc32c(crc32c, cp_results[i].crc32c, component_lengths[i])
    bytes_transferred = 0
    expect_gzip = ObjectIsGzipEncoded(src_obj_metadata)
    for cp_result in cp_results:
        PutToQueueWithTimeout(status_queue, FileMessage(src_url, dst_url, time.time(), size=cp_result.component_total_size, finished=True, component_num=cp_result.component_num, message_type=FileMessage.COMPONENT_TO_DOWNLOAD))
        bytes_transferred += cp_result.bytes_transferred
        server_gzip = cp_result.server_encoding and cp_result.server_encoding.lower().endswith('gzip')
        if server_gzip and (not expect_gzip):
            raise CommandException('Download of %s failed because the server sent back data with an unexpected encoding.' % dst_url.object_name)
    return (bytes_transferred, crc32c)
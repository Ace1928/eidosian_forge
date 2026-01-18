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
def _DoParallelCompositeUpload(fp, src_url, dst_url, dst_obj_metadata, canned_acl, file_size, preconditions, gsutil_api, command_obj, copy_exception_handler, logger, gzip_encoded=False):
    """Uploads a local file to a cloud object using parallel composite upload.

  The file is partitioned into parts, and then the parts are uploaded in
  parallel, composed to form the original destination object, and deleted.

  Args:
    fp: The file object to be uploaded.
    src_url: FileUrl representing the local file.
    dst_url: CloudUrl representing the destination file.
    dst_obj_metadata: apitools Object describing the destination object.
    canned_acl: The canned acl to apply to the object, if any.
    file_size: The size of the source file in bytes.
    preconditions: Cloud API Preconditions for the final object.
    gsutil_api: gsutil Cloud API instance to use.
    command_obj: Command object (for calling Apply).
    copy_exception_handler: Copy exception handler (for use in Apply).
    logger: logging.Logger for outputting log messages.
    gzip_encoded: Whether to use gzip transport encoding for the upload.

  Returns:
    Elapsed upload time, uploaded Object with generation, crc32c, and size
    fields populated.
  """
    start_time = time.time()
    dst_bucket_url = StorageUrlFromString(dst_url.bucket_url_string)
    api_selector = gsutil_api.GetApiSelector(provider=dst_url.scheme)
    encryption_keywrapper = GetEncryptionKeyWrapper(config)
    encryption_key_sha256 = encryption_keywrapper.crypto_key_sha256 if encryption_keywrapper else None
    tracker_file_name = GetTrackerFilePath(dst_url, TrackerFileType.PARALLEL_UPLOAD, api_selector, src_url)
    existing_enc_key_sha256, existing_prefix, existing_components = ReadParallelUploadTrackerFile(tracker_file_name, logger)
    existing_prefix, existing_components = ValidateParallelCompositeTrackerData(tracker_file_name, existing_enc_key_sha256, existing_prefix, existing_components, encryption_key_sha256, dst_bucket_url, command_obj, logger, _DeleteTempComponentObjectFn, _RmExceptionHandler)
    encryption_key_sha256 = encryption_key_sha256.decode('ascii') if encryption_key_sha256 is not None else None
    random_prefix = existing_prefix if existing_prefix is not None else GenerateComponentObjectPrefix(encryption_key_sha256=encryption_key_sha256)
    WriteParallelUploadTrackerFile(tracker_file_name, random_prefix, existing_components, encryption_key_sha256=encryption_key_sha256)
    tracker_file_lock = parallelism_framework_util.CreateLock()
    components_info = {}
    dst_args = _PartitionFile(canned_acl, dst_obj_metadata.contentType, dst_bucket_url, file_size, fp, random_prefix, src_url, dst_obj_metadata.storageClass, tracker_file_name, tracker_file_lock, encryption_key_sha256=encryption_key_sha256, gzip_encoded=gzip_encoded)
    components_to_upload, existing_components, existing_objects_to_delete = FilterExistingComponents(dst_args, existing_components, dst_bucket_url, gsutil_api)
    for component in components_to_upload:
        components_info[component.dst_url.url_string] = (FileMessage.COMPONENT_TO_UPLOAD, component.file_length)
        PutToQueueWithTimeout(gsutil_api.status_queue, FileMessage(component.src_url, component.dst_url, time.time(), size=component.file_length, finished=False, component_num=_GetComponentNumber(component.dst_url), message_type=FileMessage.COMPONENT_TO_UPLOAD))
    for component in existing_components:
        component_str = component[0].versionless_url_string
        components_info[component_str] = (FileMessage.EXISTING_COMPONENT, component[1])
        PutToQueueWithTimeout(gsutil_api.status_queue, FileMessage(src_url, component[0], time.time(), finished=False, size=component[1], component_num=_GetComponentNumber(component[0]), message_type=FileMessage.EXISTING_COMPONENT))
    for component in existing_objects_to_delete:
        PutToQueueWithTimeout(gsutil_api.status_queue, FileMessage(src_url, component, time.time(), finished=False, message_type=FileMessage.EXISTING_OBJECT_TO_DELETE))
    cp_results = command_obj.Apply(_PerformParallelUploadFileToObject, components_to_upload, copy_exception_handler, ('op_failure_count', 'total_bytes_transferred'), arg_checker=gslib.command.DummyArgChecker, parallel_operations_override=command_obj.ParallelOverrideReason.SLICE, should_return_results=True)
    uploaded_components = []
    for cp_result in cp_results:
        uploaded_components.append(cp_result[2])
    components = uploaded_components + [i[0] for i in existing_components]
    if len(components) == len(dst_args):
        components = sorted(components, key=_GetComponentNumber)
        request_components = []
        for component_url in components:
            src_obj_metadata = apitools_messages.ComposeRequest.SourceObjectsValueListEntry(name=component_url.object_name)
            if component_url.HasGeneration():
                src_obj_metadata.generation = long(component_url.generation)
            request_components.append(src_obj_metadata)
        composed_object = gsutil_api.ComposeObject(request_components, dst_obj_metadata, preconditions=preconditions, provider=dst_url.scheme, fields=['crc32c', 'generation', 'size'], encryption_tuple=encryption_keywrapper)
        try:
            objects_to_delete = components + existing_objects_to_delete
            command_obj.Apply(_DeleteTempComponentObjectFn, objects_to_delete, _RmExceptionHandler, arg_checker=gslib.command.DummyArgChecker, parallel_operations_override=command_obj.ParallelOverrideReason.SLICE)
            for component in components:
                component_str = component.versionless_url_string
                try:
                    PutToQueueWithTimeout(gsutil_api.status_queue, FileMessage(src_url, component, time.time(), finished=True, component_num=_GetComponentNumber(component), size=components_info[component_str][1], message_type=components_info[component_str][0]))
                except:
                    PutToQueueWithTimeout(gsutil_api.status_queue, FileMessage(src_url, component, time.time(), finished=True))
            for component in existing_objects_to_delete:
                PutToQueueWithTimeout(gsutil_api.status_queue, FileMessage(src_url, component, time.time(), finished=True, message_type=FileMessage.EXISTING_OBJECT_TO_DELETE))
        except Exception:
            logger.warn('Failed to delete some of the following temporary objects:\n' + '\n'.join(dst_args.keys()))
        finally:
            with tracker_file_lock:
                DeleteTrackerFile(tracker_file_name)
    else:
        raise CommandException('Some temporary components were not uploaded successfully. Please retry this upload.')
    elapsed_time = time.time() - start_time
    return (elapsed_time, composed_object)
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
import datetime
import errno
import json
import os
import pickle
import random
import re
import socket
import tempfile
import textwrap
import threading
import time
import xml
from xml.dom.minidom import parseString as XmlParseString
from xml.sax import _exceptions as SaxExceptions
import six
from six.moves import http_client
import boto
from boto import config
from boto import handler
from boto.gs.cors import Cors
from boto.gs.lifecycle import LifecycleConfig
from boto.s3.cors import CORSConfiguration as S3Cors
from boto.s3.deletemarker import DeleteMarker
from boto.s3.lifecycle import Lifecycle as S3Lifecycle
from boto.s3.prefix import Prefix
from boto.s3.tagging import Tags
import boto.exception
import boto.utils
from gslib.boto_resumable_upload import BotoResumableUpload
from gslib.cloud_api import AccessDeniedException
from gslib.cloud_api import ArgumentException
from gslib.cloud_api import BadRequestException
from gslib.cloud_api import CloudApi
from gslib.cloud_api import NotEmptyException
from gslib.cloud_api import NotFoundException
from gslib.cloud_api import PreconditionException
from gslib.cloud_api import ResumableDownloadException
from gslib.cloud_api import ResumableUploadAbortException
from gslib.cloud_api import ResumableUploadException
from gslib.cloud_api import ResumableUploadStartOverException
from gslib.cloud_api import ServiceException
import gslib.devshell_auth_plugin  # pylint: disable=unused-import
from gslib.exception import CommandException
from gslib.exception import InvalidUrlError
from gslib.project_id import GOOG_PROJ_ID_HDR
from gslib.project_id import PopulateProjectId
from gslib.storage_url import GenerationFromUrlAndString
from gslib.storage_url import StorageUrlFromString
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import parallelism_framework_util
from gslib.utils.boto_util import ConfigureNoOpAuthIfNeeded
from gslib.utils.boto_util import GetMaxRetryDelay
from gslib.utils.boto_util import GetNumRetries
from gslib.utils.cloud_api_helper import ListToGetFields
from gslib.utils.cloud_api_helper import ValidateDstObjectMetadata
from gslib.utils.constants import DEFAULT_FILE_BUFFER_SIZE
from gslib.utils.constants import REQUEST_REASON_ENV_VAR
from gslib.utils.constants import REQUEST_REASON_HEADER_KEY
from gslib.utils.constants import S3_DELETE_MARKER_GUID
from gslib.utils.constants import UTF8
from gslib.utils.constants import XML_PROGRESS_CALLBACKS
from gslib.utils.hashing_helper import Base64EncodeHash
from gslib.utils.hashing_helper import Base64ToHexHash
from gslib.utils.metadata_util import AddAcceptEncodingGzipIfNeeded
from gslib.utils.parallelism_framework_util import multiprocessing_context
from gslib.utils.text_util import EncodeStringAsLong
from gslib.utils.translation_helper import AclTranslation
from gslib.utils.translation_helper import AddS3MarkerAclToObjectMetadata
from gslib.utils.translation_helper import CorsTranslation
from gslib.utils.translation_helper import CreateBucketNotFoundException
from gslib.utils.translation_helper import CreateNotFoundExceptionForObjectWrite
from gslib.utils.translation_helper import CreateObjectNotFoundException
from gslib.utils.translation_helper import DEFAULT_CONTENT_TYPE
from gslib.utils.translation_helper import HeadersFromObjectMetadata
from gslib.utils.translation_helper import LabelTranslation
from gslib.utils.translation_helper import LifecycleTranslation
from gslib.utils.translation_helper import REMOVE_CORS_CONFIG
from gslib.utils.translation_helper import S3MarkerAclFromObjectMetadata
from gslib.utils.translation_helper import UnaryDictToXml
from gslib.utils.unit_util import TWO_MIB
def _TranslateBotoException(self, e, bucket_name=None, object_name=None, generation=None, not_found_exception=None):
    """Translates boto exceptions into their gsutil Cloud API equivalents.

    Args:
      e: Any exception in TRANSLATABLE_BOTO_EXCEPTIONS.
      bucket_name: Optional bucket name in request that caused the exception.
      object_name: Optional object name in request that caused the exception.
      generation: Optional generation in request that caused the exception.
      not_found_exception: Optional exception to raise in the not-found case.

    Returns:
      ServiceException for translatable exceptions, None
      otherwise.

    Because we're using isinstance, check for subtypes first.
    """
    if isinstance(e, boto.exception.StorageResponseError):
        if e.status == 400:
            return BadRequestException(e.code, status=e.status, body=e.body)
        elif e.status == 401 or e.status == 403:
            return AccessDeniedException(e.code, status=e.status, body=e.body)
        elif e.status == 404:
            if not_found_exception:
                setattr(not_found_exception, 'status', e.status)
                return not_found_exception
            elif bucket_name:
                if object_name:
                    return CreateObjectNotFoundException(e.status, self.provider, bucket_name, object_name, generation=generation)
                return CreateBucketNotFoundException(e.status, self.provider, bucket_name)
            return NotFoundException(e.message, status=e.status, body=e.body)
        elif e.status == 409 and e.code and ('BucketNotEmpty' in e.code):
            return NotEmptyException('BucketNotEmpty (%s)' % bucket_name, status=e.status, body=e.body)
        elif e.status == 410:
            return ResumableUploadStartOverException(e.message)
        elif e.status == 412:
            return PreconditionException(e.code, status=e.status, body=e.body)
    if isinstance(e, boto.exception.StorageCreateError):
        return ServiceException('Bucket already exists.', status=e.status, body=e.body)
    if isinstance(e, boto.exception.BotoServerError):
        return ServiceException(e.message, status=e.status, body=e.body)
    if isinstance(e, boto.exception.InvalidUriError):
        if e.message and NON_EXISTENT_OBJECT_REGEX.match(e.message):
            return NotFoundException(e.message, status=404)
        return InvalidUrlError(e.message)
    if isinstance(e, boto.exception.ResumableUploadException):
        if e.disposition == boto.exception.ResumableTransferDisposition.ABORT:
            return ResumableUploadAbortException(e.message)
        elif e.disposition == boto.exception.ResumableTransferDisposition.START_OVER:
            return ResumableUploadStartOverException(e.message)
        else:
            return ResumableUploadException(e.message)
    if isinstance(e, boto.exception.ResumableDownloadException):
        return ResumableDownloadException(e.message)
    return None
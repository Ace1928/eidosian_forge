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
def _UploadSetup(self, object_metadata, preconditions=None):
    """Shared upload implementation.

    Args:
      object_metadata: Object metadata describing destination object.
      preconditions: Optional gsutil Cloud API preconditions.

    Returns:
      Headers dictionary, StorageUri for upload (based on inputs)
    """
    ValidateDstObjectMetadata(object_metadata)
    headers = self._CreateBaseHeaders()
    headers.update(HeadersFromObjectMetadata(object_metadata, self.provider))
    if object_metadata.crc32c:
        if 'x-goog-hash' in headers:
            headers['x-goog-hash'] += ',crc32c=%s' % object_metadata.crc32c.rstrip('\n')
        else:
            headers['x-goog-hash'] = 'crc32c=%s' % object_metadata.crc32c.rstrip('\n')
    if object_metadata.md5Hash:
        if 'x-goog-hash' in headers:
            headers['x-goog-hash'] += ',md5=%s' % object_metadata.md5Hash.rstrip('\n')
        else:
            headers['x-goog-hash'] = 'md5=%s' % object_metadata.md5Hash.rstrip('\n')
    if 'content-type' in headers and (not headers['content-type']):
        headers['content-type'] = 'application/octet-stream'
    self._AddPreconditionsToHeaders(preconditions, headers)
    dst_uri = self._StorageUriForObject(object_metadata.bucket, object_metadata.name)
    return (headers, dst_uri)
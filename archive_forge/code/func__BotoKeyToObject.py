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
def _BotoKeyToObject(self, key, fields=None):
    """Constructs an apitools Object from a boto key.

    Args:
      key: Boto key to construct Object from.
      fields: If present, construct the apitools Object with only this set of
              metadata fields.

    Returns:
      apitools Object corresponding to key.
    """
    custom_metadata = None
    if not fields or 'metadata' in fields or len([field for field in fields if field.startswith('metadata/')]) >= 1:
        custom_metadata = self._TranslateBotoKeyCustomMetadata(key)
    cache_control = None
    if not fields or 'cacheControl' in fields:
        cache_control = getattr(key, 'cache_control', None)
    component_count = None
    if not fields or 'componentCount' in fields:
        component_count = getattr(key, 'component_count', None)
    content_disposition = None
    if not fields or 'contentDisposition' in fields:
        content_disposition = getattr(key, 'content_disposition', None)
    generation = self._TranslateBotoKeyGeneration(key)
    metageneration = None
    if not fields or 'metageneration' in fields:
        metageneration = self._TranslateBotoKeyMetageneration(key)
    time_created = None
    if not fields or 'timeCreated' in fields:
        time_created = self._TranslateBotoKeyTimestamp(key)
    etag = None
    if not fields or 'etag' in fields:
        etag = getattr(key, 'etag', None)
        if etag:
            etag = etag.strip('"\'')
    crc32c = None
    if not fields or 'crc32c' in fields:
        if hasattr(key, 'cloud_hashes') and 'crc32c' in key.cloud_hashes:
            crc32c = base64.b64encode(key.cloud_hashes['crc32c']).rstrip(b'\n')
    md5_hash = None
    if not fields or 'md5Hash' in fields:
        if hasattr(key, 'cloud_hashes') and 'md5' in key.cloud_hashes:
            md5_hash = base64.b64encode(key.cloud_hashes['md5']).rstrip(b'\n')
        elif self._GetMD5FromETag(getattr(key, 'etag', None)):
            md5_hash = Base64EncodeHash(self._GetMD5FromETag(key.etag))
        elif self.provider == 's3':
            self.logger.warn('Non-MD5 etag (%s) present for key %s, data integrity checks are not possible.', key.etag, key)
    media_link = None
    if not fields or 'mediaLink' in fields:
        media_link = binascii.b2a_base64(pickle.dumps(key, pickle.HIGHEST_PROTOCOL))
    size = None
    if not fields or 'size' in fields:
        size = key.size or 0
    storage_class = None
    if not fields or 'storageClass' in fields:
        storage_class = getattr(key, '_storage_class', None)
    if six.PY3:
        if crc32c and isinstance(crc32c, bytes):
            crc32c = crc32c.decode('ascii')
        if md5_hash and isinstance(md5_hash, bytes):
            md5_hash = md5_hash.decode('ascii')
    cloud_api_object = apitools_messages.Object(bucket=key.bucket.name, name=key.name, size=size, contentEncoding=key.content_encoding, contentLanguage=key.content_language, contentType=key.content_type, cacheControl=cache_control, contentDisposition=content_disposition, etag=etag, crc32c=crc32c, md5Hash=md5_hash, generation=generation, metageneration=metageneration, componentCount=component_count, timeCreated=time_created, metadata=custom_metadata, mediaLink=media_link, storageClass=storage_class)
    self._TranslateDeleteMarker(key, cloud_api_object)
    if not fields or 'acl' in fields:
        generation_str = GenerationFromUrlAndString(StorageUrlFromString(self.provider), generation)
        self._TranslateBotoKeyAcl(key, cloud_api_object, generation=generation_str)
    return cloud_api_object
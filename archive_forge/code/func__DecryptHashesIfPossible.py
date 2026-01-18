from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from contextlib import contextmanager
import functools
from six.moves import http_client
import json
import logging
import os
import socket
import ssl
import time
import traceback
import six
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import http_wrapper as apitools_http_wrapper
from apitools.base.py import transfer as apitools_transfer
from apitools.base.py.util import CalculateWaitForRetry
from boto import config
import httplib2
import oauth2client
from gslib import context_config
from gslib.cloud_api import AccessDeniedException
from gslib.cloud_api import ArgumentException
from gslib.cloud_api import BadRequestException
from gslib.cloud_api import CloudApi
from gslib.cloud_api import EncryptionException
from gslib.cloud_api import NotEmptyException
from gslib.cloud_api import NotFoundException
from gslib.cloud_api import PreconditionException
from gslib.cloud_api import Preconditions
from gslib.cloud_api import PublishPermissionDeniedException
from gslib.cloud_api import ResumableDownloadException
from gslib.cloud_api import ResumableUploadAbortException
from gslib.cloud_api import ResumableUploadException
from gslib.cloud_api import ResumableUploadStartOverException
from gslib.cloud_api import ServiceException
from gslib.exception import CommandException
from gslib.gcs_json_credentials import SetUpJsonCredentialsAndCache
from gslib.gcs_json_media import BytesTransferredContainer
from gslib.gcs_json_media import DownloadCallbackConnectionClassFactory
from gslib.gcs_json_media import HttpWithDownloadStream
from gslib.gcs_json_media import HttpWithNoRetries
from gslib.gcs_json_media import UploadCallbackConnectionClassFactory
from gslib.gcs_json_media import WrapDownloadHttpRequest
from gslib.gcs_json_media import WrapUploadHttpRequest
from gslib.impersonation_credentials import ImpersonationCredentials
from gslib.no_op_credentials import NoOpCredentials
from gslib.progress_callback import ProgressCallbackWithTimeout
from gslib.project_id import PopulateProjectId
from gslib.third_party.storage_apitools import storage_v1_client as apitools_client
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.tracker_file import DeleteTrackerFile
from gslib.tracker_file import GetRewriteTrackerFilePath
from gslib.tracker_file import HashRewriteParameters
from gslib.tracker_file import ReadRewriteTrackerFile
from gslib.tracker_file import WriteRewriteTrackerFile
from gslib.utils.boto_util import GetCertsFile
from gslib.utils.boto_util import GetGcsJsonApiVersion
from gslib.utils.boto_util import GetJsonResumableChunkSize
from gslib.utils.boto_util import GetMaxRetryDelay
from gslib.utils.boto_util import GetNewHttp
from gslib.utils.boto_util import GetNumRetries
from gslib.utils.boto_util import JsonResumableChunkSizeDefined
from gslib.utils.cloud_api_helper import ListToGetFields
from gslib.utils.cloud_api_helper import ValidateDstObjectMetadata
from gslib.utils.constants import IAM_POLICY_VERSION
from gslib.utils.constants import NUM_OBJECTS_PER_LIST_PAGE
from gslib.utils.constants import REQUEST_REASON_ENV_VAR
from gslib.utils.constants import REQUEST_REASON_HEADER_KEY
from gslib.utils.constants import UTF8
from gslib.utils.encryption_helper import Base64Sha256FromBase64EncryptionKey
from gslib.utils.encryption_helper import CryptoKeyType
from gslib.utils.encryption_helper import CryptoKeyWrapperFromKey
from gslib.utils.encryption_helper import FindMatchingCSEKInBotoConfig
from gslib.utils.metadata_util import AddAcceptEncodingGzipIfNeeded
from gslib.utils.retry_util import LogAndHandleRetries
from gslib.utils.signurl_helper import CreatePayload, GetFinalUrl
from gslib.utils.text_util import GetPrintableExceptionString
from gslib.utils.translation_helper import CreateBucketNotFoundException
from gslib.utils.translation_helper import CreateNotFoundExceptionForObjectWrite
from gslib.utils.translation_helper import CreateObjectNotFoundException
from gslib.utils.translation_helper import DEFAULT_CONTENT_TYPE
from gslib.utils.translation_helper import PRIVATE_DEFAULT_OBJ_ACL
from gslib.utils.translation_helper import REMOVE_CORS_CONFIG
from oauth2client.service_account import ServiceAccountCredentials
def _DecryptHashesIfPossible(self, bucket_name, object_metadata, fields=None):
    """Attempts to decrypt object metadata.

    Helps fetch encrypted object metadata fields (if requested) for API method
    calls that requested them but did not receive them; this occurs for
    ListObjects(), along with GetObjectMetadata() calls for which the object is
    CSEK-encrypted but no decryption information was supplied.

    This method issues a StorageObjectsGetRequest (if object hash fields were
    requested but not retrieved), formatted appropriately depending on the
    encryption method.

    Args:
      bucket_name: String of bucket containing the object.
      object_metadata: apitools Object describing the object. Must include
          name, generation, and customerEncryption fields.
      fields: ListObjects-format fields to return.

    Returns:
      - apitools_messages.Object with decrypted hash fields if decryption was
      performed.
      - None if the object was not encrypted, or CSEK-encrypted but we could not
      find a matching decryption key.
      - gcs_json_api._SKIP_LISTING_OBJECT if the object no longer exists.
    """
    get_fields = ListToGetFields(list_fields=fields)
    if not self._ObjectIsEncrypted(object_metadata):
        self._RaiseIfNoDesiredHashesPresentForUnencryptedObject(object_metadata, fields=get_fields)
        return None
    get_metadata_func = None
    if self._ObjectKMSEncryptedAndNeedHashes(object_metadata, fields=get_fields):
        get_metadata_func = functools.partial(self._GetObjectMetadataHelper, bucket_name, object_metadata.name, generation=object_metadata.generation, fields=get_fields)
    elif self._ObjectCSEKEncryptedAndNeedHashes(object_metadata, fields=get_fields):
        key_sha256 = object_metadata.customerEncryption.keySha256
        if six.PY3:
            if not isinstance(key_sha256, bytes):
                key_sha256 = key_sha256.encode('ascii')
        decryption_key = FindMatchingCSEKInBotoConfig(key_sha256, config)
        if decryption_key:
            get_metadata_func = functools.partial(self._GetObjectMetadataHelper, bucket_name, object_metadata.name, generation=object_metadata.generation, fields=get_fields, decryption_tuple=CryptoKeyWrapperFromKey(decryption_key))
    if get_metadata_func is None:
        return
    try:
        return get_metadata_func()
    except NotFoundException:
        return _SKIP_LISTING_OBJECT
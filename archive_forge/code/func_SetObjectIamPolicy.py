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
def SetObjectIamPolicy(self, bucket_name, object_name, policy, generation=None, provider=None):
    if generation is not None:
        generation = long(generation)
    api_request = apitools_messages.StorageObjectsSetIamPolicyRequest(bucket=bucket_name, object=object_name, generation=generation, policy=policy, userProject=self.user_project)
    global_params = apitools_messages.StandardQueryParameters()
    try:
        return self.api_client.objects.SetIamPolicy(api_request, global_params=global_params)
    except TRANSLATABLE_APITOOLS_EXCEPTIONS as e:
        self._TranslateExceptionAndRaise(e, bucket_name=bucket_name, object_name=object_name)
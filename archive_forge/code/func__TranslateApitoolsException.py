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
def _TranslateApitoolsException(self, e, bucket_name=None, object_name=None, generation=None, not_found_exception=None):
    """Translates apitools exceptions into their gsutil Cloud Api equivalents.

    Args:
      e: Any exception in TRANSLATABLE_APITOOLS_EXCEPTIONS.
      bucket_name: Optional bucket name in request that caused the exception.
      object_name: Optional object name in request that caused the exception.
      generation: Optional generation in request that caused the exception.
      not_found_exception: Optional exception to raise in the not-found case.

    Returns:
      ServiceException for translatable exceptions, None
      otherwise.
    """
    if isinstance(e, apitools_exceptions.HttpError):
        message = self._GetMessageFromHttpError(e)
        if e.status_code == 400:
            return BadRequestException(message or 'Bad Request', status=e.status_code)
        elif e.status_code == 401:
            if 'Login Required' in str(e):
                return AccessDeniedException(message or 'Access denied: login required.', status=e.status_code)
            elif 'insufficient_scope' in str(e):
                return AccessDeniedException(_INSUFFICIENT_OAUTH2_SCOPE_MESSAGE, status=e.status_code, body=self._GetAcceptableScopesFromHttpError(e))
        elif e.status_code == 403:
            if 'The account for the specified project has been disabled' in str(e):
                return AccessDeniedException(message or 'Account disabled.', status=e.status_code)
            elif 'Daily Limit for Unauthenticated Use Exceeded' in str(e):
                return AccessDeniedException(message or 'Access denied: quota exceeded. Is your project ID valid?', status=e.status_code)
            elif 'The bucket you tried to delete was not empty.' in str(e):
                return NotEmptyException('BucketNotEmpty (%s)' % bucket_name, status=e.status_code)
            elif 'The bucket you tried to create requires domain ownership verification.' in str(e):
                return AccessDeniedException('The bucket you tried to create requires domain ownership verification. Please see https://cloud.google.com/storage/docs/naming?hl=en#verification for more details.', status=e.status_code)
            elif 'User Rate Limit Exceeded' in str(e):
                return AccessDeniedException('Rate limit exceeded. Please retry this request later.', status=e.status_code)
            elif 'Access Not Configured' in str(e):
                return AccessDeniedException('Access Not Configured. Please go to the Google Cloud Platform Console (https://cloud.google.com/console#/project) for your project, select APIs and Auth and enable the Google Cloud Storage JSON API.', status=e.status_code)
            elif 'insufficient_scope' in str(e):
                return AccessDeniedException(_INSUFFICIENT_OAUTH2_SCOPE_MESSAGE, status=e.status_code, body=self._GetAcceptableScopesFromHttpError(e))
            elif 'does not have permission to publish messages' in str(e):
                return PublishPermissionDeniedException(message, status=e.status_code)
            else:
                return AccessDeniedException(message or str(e), status=e.status_code)
        elif e.status_code == 404:
            if not_found_exception:
                setattr(not_found_exception, 'status', e.status_code)
                return not_found_exception
            elif bucket_name:
                if object_name:
                    return CreateObjectNotFoundException(e.status_code, self.provider, bucket_name, object_name, generation=generation)
                return CreateBucketNotFoundException(e.status_code, self.provider, bucket_name)
            return NotFoundException(message or e.message, status=e.status_code)
        elif e.status_code == 409 and bucket_name:
            if 'The bucket you tried to delete is not empty.' in str(e):
                return NotEmptyException('BucketNotEmpty (%s)' % bucket_name, status=e.status_code)
            return ServiceException("A Cloud Storage bucket named '%s' already exists. Try another name. Bucket names must be globally unique across all Google Cloud projects, including those outside of your organization." % bucket_name, status=e.status_code)
        elif e.status_code == 412:
            return PreconditionException(message, status=e.status_code)
        return ServiceException(message, status=e.status_code)
    elif isinstance(e, apitools_exceptions.TransferInvalidError):
        return ServiceException('Transfer invalid (possible encoding error: %s)' % str(e))
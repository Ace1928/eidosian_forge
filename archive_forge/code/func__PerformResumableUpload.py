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
def _PerformResumableUpload(self, upload_stream, authorized_upload_http, content_type, size, serialization_data, apitools_strategy, apitools_request, global_params, bytes_uploaded_container, tracker_callback, addl_headers, progress_callback, gzip_encoded):
    try:
        if serialization_data:
            apitools_upload = apitools_transfer.Upload.FromData(upload_stream, serialization_data, self.api_client.http, num_retries=self.num_retries, gzip_encoded=gzip_encoded, client=self.api_client)
            apitools_upload.chunksize = GetJsonResumableChunkSize()
            apitools_upload.bytes_http = authorized_upload_http
        else:
            apitools_upload = apitools_transfer.Upload(upload_stream, content_type, total_size=size, chunksize=GetJsonResumableChunkSize(), auto_transfer=False, num_retries=self.num_retries, gzip_encoded=gzip_encoded)
            apitools_upload.strategy = apitools_strategy
            apitools_upload.bytes_http = authorized_upload_http
            with self._ApitoolsRequestHeaders(addl_headers):
                self.api_client.objects.Insert(apitools_request, upload=apitools_upload, global_params=global_params)
        apitools_upload.retry_func = LogAndHandleRetries(is_data_transfer=True, status_queue=self.status_queue)

        def _NoOpCallback(unused_response, unused_upload_object):
            pass
        bytes_uploaded_container.bytes_transferred = apitools_upload.progress
        if tracker_callback:
            tracker_callback(json.dumps(apitools_upload.serialization_data))
        retries = 0
        last_progress_byte = apitools_upload.progress
        while retries <= self.num_retries:
            try:
                if not gzip_encoded and size and (not JsonResumableChunkSizeDefined()):
                    http_response = apitools_upload.StreamMedia(callback=_NoOpCallback, finish_callback=_NoOpCallback, additional_headers=addl_headers)
                else:
                    http_response = apitools_upload.StreamInChunks(callback=_NoOpCallback, finish_callback=_NoOpCallback, additional_headers=addl_headers)
                processed_response = self.api_client.objects.ProcessHttpResponse(self.api_client.objects.GetMethodConfig('Insert'), http_response)
                if size is None and progress_callback:
                    progress_callback(apitools_upload.total_size, apitools_upload.total_size)
                return processed_response
            except HTTP_TRANSFER_EXCEPTIONS as e:
                self._ValidateHttpAccessTokenRefreshError(e)
                apitools_http_wrapper.RebuildHttpConnections(apitools_upload.bytes_http)
                while retries <= self.num_retries:
                    try:
                        apitools_upload.RefreshResumableUploadState()
                        start_byte = apitools_upload.progress
                        bytes_uploaded_container.bytes_transferred = start_byte
                        break
                    except HTTP_TRANSFER_EXCEPTIONS as e2:
                        self._ValidateHttpAccessTokenRefreshError(e2)
                        apitools_http_wrapper.RebuildHttpConnections(apitools_upload.bytes_http)
                        retries += 1
                        if retries > self.num_retries:
                            raise ResumableUploadException('Transfer failed after %d retries. Final exception: %s' % (self.num_retries, e2))
                        time.sleep(CalculateWaitForRetry(retries, max_wait=self.max_retry_wait))
                if start_byte > last_progress_byte:
                    last_progress_byte = start_byte
                    retries = 0
                else:
                    retries += 1
                    if retries > self.num_retries:
                        raise ResumableUploadException('Transfer failed after %d retries. Final exception: %s' % (self.num_retries, GetPrintableExceptionString(e)))
                    time.sleep(CalculateWaitForRetry(retries, max_wait=self.max_retry_wait))
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug('Retrying upload from byte %s after exception: %s. Trace: %s', start_byte, GetPrintableExceptionString(e), traceback.format_exc())
    except TRANSLATABLE_APITOOLS_EXCEPTIONS as e:
        resumable_ex = self._TranslateApitoolsResumableUploadException(e)
        if resumable_ex:
            raise resumable_ex
        else:
            raise
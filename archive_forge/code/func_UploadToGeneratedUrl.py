from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import random
import re
import string
import time
from typing import Dict, Optional
from apitools.base.py import exceptions as http_exceptions
from apitools.base.py import http_wrapper
from apitools.base.py import transfer
from apitools.base.py import util as http_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.functions import exceptions
from googlecloudsdk.command_lib.util import gcloudignore
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core import transports
from googlecloudsdk.core.util import archive
from googlecloudsdk.core.util import files as file_utils
import six
from six.moves import http_client
from six.moves import range
def UploadToGeneratedUrl(source_zip: str, url: str, extra_headers: Optional[Dict[str, str]]=None) -> None:
    """Upload the given source ZIP file to provided generated URL.

  Args:
    source_zip: the source ZIP file to upload.
    url: the signed Cloud Storage URL to upload to.
    extra_headers: extra headers to attach to the request.
  """
    extra_headers = extra_headers or {}
    upload = transfer.Upload.FromFile(source_zip, mime_type='application/zip')

    def _UploadRetryFunc(retry_args: http_wrapper.ExceptionRetryArgs) -> None:
        if isinstance(retry_args.exc, http_exceptions.HttpForbiddenError):
            log.debug('Caught delayed permission propagation error, retrying')
            http_wrapper.RebuildHttpConnections(retry_args.http)
            time.sleep(http_util.CalculateWaitForRetry(retry_args.num_retries, max_wait=retry_args.max_retry_wait))
        else:
            upload.retry_func(retry_args)
    try:
        upload_request = http_wrapper.Request(url, http_method='PUT', headers={'content-type': 'application/zip', **extra_headers})
        upload_request.body = upload.stream.read()
        response = http_wrapper.MakeRequest(transports.GetApitoolsTransport(), upload_request, retry_func=_UploadRetryFunc, check_response_func=_UploadFileToGeneratedUrlCheckResponse, retries=upload.num_retries)
    finally:
        upload.stream.close()
    if response.status_code // 100 != 2:
        raise exceptions.SourceUploadError('Failed to upload the function source code to signed url: {url}. Status: [{code}:{detail}]'.format(url=url, code=response.status_code, detail=response.content))
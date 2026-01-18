from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import batch
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.core import properties
from six.moves.urllib import parse
def AsyncRequests(self, requests, errors_to_collect=None):
    """Issues async request for given set of requests.

    Return immediately without waiting for the operation in progress to complete

    Args:
      requests: list(tuple(service, method, payload)), where service is
        apitools.base.py.base_api.BaseApiService, method is str, method name,
        e.g. 'Get', 'CreateInstance', payload is a subclass of
        apitools.base.protorpclite.messages.Message.
      errors_to_collect: list, output only, can be None, contains instances of
        api_exceptions.HttpException for each request with exception.

    Returns:
      list of responses, matching list of requests. Some responses can be
        errors.
    """
    if not _ForceBatchRequest() and len(requests) == 1:
        responses = []
        errors = errors_to_collect if errors_to_collect is not None else []
        service, method, request_body = requests[0]
        num_retries = service.client.num_retries
        service.client.num_retries = 0
        try:
            response = getattr(service, method)(request=request_body)
            responses.append(response)
        except apitools_exceptions.HttpError as exception:
            errors.append(api_exceptions.HttpException(exception))
            responses.append(None)
        except apitools_exceptions.Error as exception:
            if hasattr(exception, 'message'):
                errors.append(Error(exception.message))
            else:
                errors.append(Error(exception))
            responses.append(None)
        service.client.num_retries = num_retries
        return responses
    else:
        batch_request = batch.BatchApiRequest(batch_url=self._batch_url)
        for service, method, request in requests:
            batch_request.Add(service, method, request)
        payloads = batch_request.Execute(self._client.http, max_batch_size=_BATCH_SIZE_LIMIT)
        responses = []
        errors = errors_to_collect if errors_to_collect is not None else []
        for payload in payloads:
            if payload.is_error:
                if isinstance(payload.exception, apitools_exceptions.HttpError):
                    errors.append(api_exceptions.HttpException(payload.exception))
                else:
                    errors.append(Error(payload.exception.message))
            responses.append(payload.response)
    return responses
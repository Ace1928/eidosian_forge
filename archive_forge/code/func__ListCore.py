from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import json
from googlecloudsdk.api_lib.compute import batch_helper
from googlecloudsdk.api_lib.compute import single_request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute import waiters
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
import six
from six.moves import zip  # pylint: disable=redefined-builtin
def _ListCore(requests, http, batch_url, errors, response_handler):
    """Makes a series of list and/or aggregatedList batch requests.

  Args:
    requests: A list of requests to make. Each element must be a 3-element tuple
      where the first element is the service, the second element is the method
      ('List' or 'AggregatedList'), and the third element is a protocol buffer
      representing either a list or aggregatedList request.
    http: An httplib2.Http-like object.
    batch_url: The handler for making batch requests.
    errors: A list for capturing errors. If any response contains an error, it
      is added to this list.
    response_handler: The function to extract information responses.

  Yields:
    Resources encapsulated in format chosen by response_handler as they are
      received from the server.
  """
    while requests:
        if not _ForceBatchRequest() and len(requests) == 1:
            service, method, request_body = requests[0]
            responses, request_errors = single_request_helper.MakeSingleRequest(service, method, request_body)
            errors.extend(request_errors)
        else:
            responses, request_errors = batch_helper.MakeRequests(requests=requests, http=http, batch_url=batch_url)
            errors.extend(request_errors)
        new_requests = []
        for i, response in enumerate(responses):
            if not response:
                continue
            service, method, request_protobuf = requests[i]
            items, next_page_token = response_handler(response, service, method, errors)
            for item in items:
                yield item
            if next_page_token:
                new_request_protobuf = copy.deepcopy(request_protobuf)
                new_request_protobuf.pageToken = next_page_token
                new_requests.append((service, method, new_request_protobuf))
        requests = new_requests
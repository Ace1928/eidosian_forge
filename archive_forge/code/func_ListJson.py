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
def ListJson(requests, http, batch_url, errors):
    """Makes a series of list and/or aggregatedList batch requests.

  This function does all of:
  - Sends batch of List/AggregatedList requests
  - Extracts items from responses
  - Handles pagination

  All requests must be sent to the same client - Compute.

  Args:
    requests: A list of requests to make. Each element must be a 3-element tuple
      where the first element is the service, the second element is the method
      ('List' or 'AggregatedList'), and the third element is a protocol buffer
      representing either a list or aggregatedList request.
    http: An httplib2.Http-like object.
    batch_url: The handler for making batch requests.
    errors: A list for capturing errors. If any response contains an error, it
      is added to this list.

  Yields:
    Resources in dicts as they are received from the server.
  """
    with requests[0][0].client.JsonResponseModel():
        for item in _ListCore(requests, http, batch_url, errors, _HandleJsonList):
            yield item
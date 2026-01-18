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
def _RequestsAreListRequests(requests):
    """Checks if all requests are of list requests."""
    list_requests = [method in ('List', 'AggregatedList', 'ListInstances', 'ListManagedInstances') for _, method, _ in requests]
    if all(list_requests):
        return True
    elif not any(list_requests):
        return False
    else:
        raise ValueError('All requests must be either list requests or non-list requests.')
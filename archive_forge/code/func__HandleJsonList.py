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
def _HandleJsonList(response, service, method, errors):
    """Extracts data from one *List response page as JSON and stores in dicts.

  Args:
    response: str, The *List response in JSON
    service: The service which responded to *List request
    method: str, Method used to list resources. One of 'List' or
      'AggregatedList'.
    errors: list, Errors from response will be appended to  this list.

  Returns:
    Pair of:
    - List of items returned in response as dicts
    - Next page token (if present, otherwise None).
  """
    items = []
    response = json.loads(response)
    if method in ('List', 'ListInstances'):
        items = response.get('items', [])
    elif method == 'ListManagedInstances':
        items = response.get('managedInstances', [])
    elif method == 'AggregatedList':
        items_field_name = service.GetMethodConfig('AggregatedList').relative_path.split('/')[-1]
        for scope_result in six.itervalues(response['items']):
            warning = scope_result.get('warning', None)
            if warning and warning['code'] == 'UNREACHABLE':
                errors.append((None, warning['message']))
            items.extend(scope_result.get(items_field_name, []))
    return (items, response.get('nextPageToken', None))
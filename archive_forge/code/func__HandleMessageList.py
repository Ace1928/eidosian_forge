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
def _HandleMessageList(response, service, method, errors):
    """Extracts data from one *List response page as Message object."""
    items = []
    if method in ('List', 'ListInstances'):
        items = response.items
    elif method == 'ListManagedInstances':
        items = response.managedInstances
    else:
        items_field_name = service.GetMethodConfig('AggregatedList').relative_path.split('/')[-1]
        for scope_result in response.items.additionalProperties:
            warning = scope_result.value.warning
            if warning and warning.code == warning.CodeValueValuesEnum.UNREACHABLE:
                errors.append((None, warning.message))
            items.extend(getattr(scope_result.value, items_field_name))
    return (items, response.nextPageToken)
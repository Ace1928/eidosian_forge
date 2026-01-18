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
def _IsEmptyOperation(operation, service):
    """Checks whether operation argument is empty.

  Args:
    operation: Operation thats checked for emptyness.
    service: Variable used to access service.client.MESSAGES_MODULE.Operation.

  Returns:
    True if operation is empty, False otherwise.
  """
    if not isinstance(operation, service.client.MESSAGES_MODULE.Operation):
        raise ValueError('operation must be instance of' + 'service.client.MESSAGES_MODULE.Operation')
    for field in operation.all_fields():
        if field.name != 'kind' and field.name != 'warnings' and (getattr(operation, field.name) is not None):
            return False
    return True
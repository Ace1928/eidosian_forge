from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import encoding
from googlecloudsdk.api_lib.services import exceptions
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.util import retry
def PrintOperation(op):
    """Print the operation.

  Args:
    op: The long running operation.

  Raises:
    OperationErrorException: if the operation fails.

  Returns:
    Nothing.
  """
    if not op.done:
        log.status.Print('Operation "{0}" is still in progress.'.format(op.name))
        return
    if op.error:
        raise exceptions.OperationErrorException('The operation "{0}" resulted in a failure "{1}".\nDetails: "{2}".'.format(op.name, op.error.message, op.error.details))
    log.status.Print('Operation "{0}" finished successfully.'.format(op.name))
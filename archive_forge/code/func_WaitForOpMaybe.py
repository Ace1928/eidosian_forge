from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ml_engine import versions_api
from googlecloudsdk.command_lib.ml_engine import models_util
from googlecloudsdk.command_lib.ml_engine import uploads
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def WaitForOpMaybe(operations_client, op, asyncronous=False, message=None):
    """Waits for an operation if asyncronous flag is on.

  Args:
    operations_client: api_lib.ml_engine.operations.OperationsClient, the client
      via which to poll
    op: Cloud ML Engine operation, the operation to poll
    asyncronous: bool, whether to wait for the operation or return immediately
    message: str, the message to display while waiting for the operation

  Returns:
    The result of the operation if asyncronous is true, or the Operation message
        otherwise
  """
    if asyncronous:
        return op
    return operations_client.WaitForOperation(op, message=message).response
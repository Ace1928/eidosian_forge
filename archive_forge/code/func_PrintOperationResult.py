from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def PrintOperationResult(op_name, op_client, service=None, wait_string='Waiting for long running operation', async_string='Asynchronous operation is in progress', is_async=False):
    """Prints results for an operation.

  Args:
    op_name: name of the operation.
    op_client: client for accessing operation data.
    service: the service which operation result can be grabbed.
    wait_string: string to use while waiting for polling operation
    async_string: string to print out for operation waiting
    is_async: whether to wait for aync operations or not.

  Returns:
    The object which is returned by the service if async is false,
    otherwise null
  """
    operation_ref = resources.REGISTRY.Parse(op_name, collection='apigateway.projects.locations.operations')
    if is_async:
        PrintOperationResultWithWaitEpilogue(operation_ref, async_string)
    else:
        return op_client.WaitForOperation(operation_ref, wait_string, service)
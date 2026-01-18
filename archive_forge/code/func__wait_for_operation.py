from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import resources
def _wait_for_operation(operation, verb, result_type=None, client=None):
    """Blocks execution until an operation completes and handles the result.

  Args:
    operation (messages.Operation): The operation to wait on.
    verb (str): The verb to use in messages, such as "delete order".
    result_type (str|none): Type of result for retrieving once operation
        completes. Will skip returning result if None.
    client (apitools.base.py.base_api.BaseApiService): API client for loading
        the results and operations clients.

  Returns:
    poller.GetResult(operation).
  Raises:
    InternalError if provided `result_type` is not `appliance` or `order`.
  """
    if client is None:
        client = apis.GetClientInstance('transferappliance', 'v1alpha1')
    operations_service = client.projects_locations_operations
    if result_type is None:
        poller = waiter.CloudOperationPollerNoResources(operations_service)
    elif result_type == 'appliance':
        poller = waiter.CloudOperationPoller(client.projects_locations_appliances, operations_service)
    elif result_type == 'order':
        poller = waiter.CloudOperationPoller(client.projects_locations_orders, operations_service)
    else:
        raise ValueError('The `result_type` must be `order`, `appliance` or None.')
    operation_ref = resources.REGISTRY.Parse(operation.name, collection=COLLECTION)
    return waiter.WaitFor(poller, operation_ref, 'Waiting for {} operation to complete'.format(verb))
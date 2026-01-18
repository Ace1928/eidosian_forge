from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai.tensorboard_time_series import client
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import resources
def ParseTensorboardOperation(operation_name):
    """Parse operation relative resource name to the operation reference object.

  Args:
    operation_name: The operation resource name

  Returns:
    The operation reference object
  """
    collection = 'aiplatform.projects.locations'
    if '/tensorboards/' in operation_name:
        collection += '.tensorboards'
    if '/experiments/' in operation_name:
        collection += '.experiments'
    if '/runs/' in operation_name:
        collection += '.runs'
    collection += '.operations'
    try:
        return resources.REGISTRY.ParseRelativeName(operation_name, collection=collection)
    except resources.WrongResourceCollectionException:
        return resources.REGISTRY.ParseRelativeName(operation_name, collection='aiplatform.projects.locations.operations')
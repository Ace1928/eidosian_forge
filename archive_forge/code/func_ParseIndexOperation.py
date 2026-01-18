from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import resources
def ParseIndexOperation(operation_name):
    """Parse operation relative resource name to the operation reference object.

  Args:
    operation_name: The operation resource name

  Returns:
    The operation reference object
  """
    if '/indexes/' in operation_name:
        try:
            return resources.REGISTRY.ParseRelativeName(operation_name, collection='aiplatform.projects.locations.indexes.operations')
        except resources.WrongResourceCollectionException:
            pass
    return resources.REGISTRY.ParseRelativeName(operation_name, collection='aiplatform.projects.locations.operations')
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def ParseOperation(operation_name):
    """Parse out an operation resource using configuration properties.

  Args:
    operation_name: str, the operation's UUID, absolute name, or relative name

  Returns:
    Operation: the parsed Operation resource
  """
    return resources.REGISTRY.Parse(operation_name, params={'projectsId': GetProject, 'locationsId': GetLocation}, collection='composer.projects.locations.operations')
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
def ParseMonitoringJobOperation(operation_name):
    """Parse operation relative resource name to the operation reference object.

  Args:
    operation_name: The operation resource name

  Returns:
    The operation reference object
  """
    if '/modelDeploymentMonitoringJobs/' in operation_name:
        try:
            return resources.REGISTRY.ParseRelativeName(operation_name, collection='aiplatform.projects.locations.modelDeploymentMonitoringJobs.operations')
        except resources.WrongResourceCollectionException:
            pass
    return resources.REGISTRY.ParseRelativeName(operation_name, collection='aiplatform.projects.locations.operations')
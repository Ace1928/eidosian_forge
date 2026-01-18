from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Optional
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.command_lib.run.integrations import integration_printer
from googlecloudsdk.command_lib.run.integrations.formatters import base
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages as runapps
def GetCallToAction(metadata: Optional[types_utils.TypeMetadata], resource: runapps.Resource, resource_status: Optional[runapps.ResourceStatus]=None):
    """Print the call to action message for the given integration.

  Args:
    metadata: the type metadata
    resource: the integration resource object
    resource_status: status of the integration

  Returns:
    A formatted string of the call to action message.
  """
    formatter = integration_printer.GetFormatter(metadata)
    return formatter.CallToAction(base.Record(name=None, metadata=metadata, region=None, resource=resource, status=resource_status, latest_deployment=None))
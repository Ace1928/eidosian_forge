from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.resourcesettings import service as settings_service
def GetValueServiceFromArgs(args):
    """Returns the value-service from the user-specified arguments.

  Args:
    args: argparse.Namespace, An object that contains the values for the
      arguments specified in the Args method.
  """
    resource_type = ComputeResourceType(args)
    return GetValueServiceFromResourceType(resource_type)
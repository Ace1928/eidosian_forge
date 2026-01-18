from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.command_lib.run.integrations.typekits import base
from googlecloudsdk.command_lib.run.integrations.typekits import custom_domains_typekit
from googlecloudsdk.command_lib.run.integrations.typekits import redis_typekit
from googlecloudsdk.command_lib.runapps import exceptions
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
def GetTypeKitByResource(resource: runapps_v1alpha1_messages.Resource) -> base.TypeKit:
    """Returns a typekit for the given resource.

  Args:
    resource: The resource object.

  Raises:
    ArgumentError: If the resource's type is not recognized.

  Returns:
    A typekit instance.
  """
    type_metadata = types_utils.GetTypeMetadataByResource(resource)
    if type_metadata is None:
        raise exceptions.ArgumentError('Integration of resource {} is not recognized'.format(resource))
    integration_type = type_metadata.integration_type
    return GetTypeKit(integration_type)
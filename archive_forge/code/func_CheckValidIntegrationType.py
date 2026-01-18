from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
from typing import List, Optional
from googlecloudsdk.command_lib.runapps import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_client
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
def CheckValidIntegrationType(integration_type: str) -> None:
    """Checks if IntegrationType is supported.

  Args:
    integration_type: integration type to validate.
  Rasies: ArgumentError
  """
    if GetTypeMetadata(integration_type) is None:
        raise exceptions.ArgumentError('Integration of type {} is not supported'.format(integration_type))
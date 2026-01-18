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
def _IntegrationVisible(integration: TypeMetadata) -> bool:
    """Returns whether or not the integration is visible.

  Args:
    integration: Each entry is defined in _INTEGRATION_TYPES

  Returns:
    True if the integration is set to visible, or if the property
      is set to true.  Otherwise it is False.
  """
    show_experimental_integrations = properties.VALUES.runapps.experimental_integrations.GetBool()
    return integration.visible or show_experimental_integrations
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Optional
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.command_lib.run.integrations import integration_printer
from googlecloudsdk.command_lib.run.integrations.formatters import base
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages as runapps
def GetSuccessMessage(integration_type, integration_name, action='deployed'):
    """Returns a user message for a successful integration deploy.

  Args:
    integration_type: str, type of the integration
    integration_name: str, name of the integration
    action: str, the action that succeeded
  """
    return '[{{bold}}{}{{reset}}] integration [{{bold}}{}{{reset}}] has been {} successfully.'.format(integration_type, integration_name, action)
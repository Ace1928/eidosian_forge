from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Optional
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.command_lib.run.integrations import integration_printer
from googlecloudsdk.command_lib.run.integrations.formatters import base
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages as runapps
def GetDeleteErrorMessage(integration_name):
    """Returns message when delete command fails.

  Args:
    integration_name: str, name of the integration.

  Returns:
    A formatted string of the error message.
  """
    return 'Deleting Integration [{}] failed, please rerun the delete command to try again.'.format(integration_name)
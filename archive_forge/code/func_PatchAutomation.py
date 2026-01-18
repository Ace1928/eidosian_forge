from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.clouddeploy import automation
from googlecloudsdk.command_lib.deploy import exceptions as cd_exceptions
from googlecloudsdk.core import resources
def PatchAutomation(resource):
    """Patches an automation resource by calling the patch automation API.

  Args:
      resource: apitools.base.protorpclite.messages.Message, automation message.

  Returns:
      The operation message.
  """
    return automation.AutomationsClient().Patch(resource)
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.clouddeploy import custom_target_type
def PatchCustomTargetType(resource):
    """Patches a custom target type resource.

  Args:
    resource: apitools.base.protorpclite.messages.Message, custom target type
      message.

  Returns:
    The operation message
  """
    return custom_target_type.CustomTargetTypesClient().Patch(resource)
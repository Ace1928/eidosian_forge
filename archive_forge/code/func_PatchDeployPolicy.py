from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.clouddeploy import deploy_policy
from googlecloudsdk.core import resources
def PatchDeployPolicy(resource):
    """Patches a deploy policy resource.

  Args:
    resource: apitools.base.protorpclite.messages.Message, deploy policy
      message.

  Returns:
    The operation message
  """
    return deploy_policy.DeployPoliciesClient().Patch(resource)
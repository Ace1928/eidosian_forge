from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.clouddeploy import client_util
from googlecloudsdk.command_lib.deploy import deploy_util
from googlecloudsdk.core import log
def CancelRollout(self, name, override_deploy_policies=None):
    """Calls the CancelRollout API to cancel a rollout.

    Args:
      name: Name of the Rollout. Format is
        projects/{project}/locations/{location}/deliveryPipelines/{deliveryPipeline}/releases/{release}/rollouts/{rollout}.
      override_deploy_policies: List of Deploy Policies to override.

    Returns:
      CancelRolloutResponse message.
    """
    if override_deploy_policies is None:
        override_deploy_policies = []
    request = self.messages.ClouddeployProjectsLocationsDeliveryPipelinesReleasesRolloutsCancelRequest(name=name, cancelRolloutRequest=self.messages.CancelRolloutRequest(overrideDeployPolicy=override_deploy_policies))
    return self._service.Cancel(request)
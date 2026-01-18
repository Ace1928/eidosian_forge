from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import flags
def ListBeta(self, location_ref):
    """Lists deployment resource pools using v1beta1 API.

    Args:
      location_ref: Resource, the parsed location to list deployment
        resource pools.

    Returns:
      Nested attribute containing list of deployment resource pools.
    """
    req = self.messages.AiplatformProjectsLocationsDeploymentResourcePoolsListRequest(parent=location_ref.RelativeName())
    return list_pager.YieldFromList(self.client.projects_locations_deploymentResourcePools, req, field='deploymentResourcePools', batch_size_attribute='pageSize')
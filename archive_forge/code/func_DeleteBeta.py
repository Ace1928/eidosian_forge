from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import flags
def DeleteBeta(self, deployment_resource_pool_ref):
    """Deletes a deployment resource pool using v1beta1 API.

    Args:
      deployment_resource_pool_ref: str, The deployment resource pool to delete.

    Returns:
      A GoogleProtobufEmpty response message for delete.
    """
    req = self.messages.AiplatformProjectsLocationsDeploymentResourcePoolsDeleteRequest(name=deployment_resource_pool_ref.RelativeName())
    operation = self.client.projects_locations_deploymentResourcePools.Delete(req)
    return operation
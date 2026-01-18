from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.clouddeploy import client_util
class DeployPoliciesClient(object):
    """Client for deploy policy service in the Cloud Deploy API."""

    def __init__(self, client=None, messages=None):
        """Initialize a deploy_policy.DeployPoliciesClient.

    Args:
      client: base_api.BaseApiClient, the client class for Cloud Deploy.
      messages: module containing the definitions of messages for Cloud Deploy.
    """
        self.client = client or client_util.GetClientInstance()
        self.messages = messages or client_util.GetMessagesModule(client)
        self._service = self.client.projects_locations_deployPolicies

    def Get(self, name):
        """Gets the deploy policy object.

    Args:
      name: deploy policy name.

    Returns:
      a deploy policy object.
    """
        request = self.messages.ClouddeployProjectsLocationsDeployPoliciesGetRequest(name=name)
        return self._service.Get(request)

    def Patch(self, obj):
        """Patches a deploy policy resource.

    Args:
      obj: apitools.base.protorpclite.messages.Message, deploy policy message.

    Returns:
      The operation message.
    """
        return self._service.Patch(self.messages.ClouddeployProjectsLocationsDeployPoliciesPatchRequest(deployPolicy=obj, allowMissing=True, name=obj.name, updateMask=DEPLOY_POLICY_UPDATE_MASK))

    def Delete(self, name):
        """Deletes a deploy policy resource.

    Args:
      name: str, deploy policy name.

    Returns:
      The operation message.
    """
        return self._service.Delete(self.messages.ClouddeployProjectsLocationsDeployPoliciesDeleteRequest(name=name, allowMissing=True))
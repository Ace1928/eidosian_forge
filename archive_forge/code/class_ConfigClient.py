from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.clouddeploy import client_util
class ConfigClient(object):
    """Client for config service in the Cloud Deploy API."""

    def __init__(self, client=None, messages=None):
        """Initialize a config.ConfigClient.

    Args:
      client: base_api.BaseApiClient, the client class for Cloud Deploy.
      messages: module containing the definitions of messages for Cloud Deploy.
    """
        self.client = client or client_util.GetClientInstance()
        self.messages = messages or client_util.GetMessagesModule(client)
        self._service = self.client.projects_locations

    def GetConfig(self, project_id, location_id):
        """Gets a config resource.

    Args:
      project_id: project id.
      location_id: region id.

    Returns:
      Config message.
    """
        return self._service.GetConfig(self.messages.ClouddeployProjectsLocationsGetConfigRequest(name='projects/{project}/locations/{location}/config'.format(project=project_id, location=location_id)))
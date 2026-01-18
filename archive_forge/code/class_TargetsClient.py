from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.clouddeploy import client_util
class TargetsClient(object):
    """Client for target service in the Cloud Deploy API."""

    def __init__(self, client=None, messages=None):
        """Initialize a target.TargetClient.

    Args:
      client: base_api.BaseApiClient, the client class for Cloud Deploy.
      messages: module containing the definitions of messages for Cloud Deploy.
    """
        self.client = client or client_util.GetClientInstance()
        self.messages = messages or client_util.GetMessagesModule(client)
        self._service = self.client.projects_locations_targets

    def Get(self, name):
        """Gets the shared target object by calling the ProjectsLocationsTargetsService.Get API.

    Args:
      name: str, target name.

    Returns:
      a target object.
    """
        request = self.messages.ClouddeployProjectsLocationsTargetsGetRequest(name=name)
        return self._service.Get(request)

    def Patch(self, target_obj):
        """Patches a target resource.

    Args:
      target_obj: apitools.base.protorpclite.messages.Message, target message.

    Returns:
      The operation message.
    """
        return self._service.Patch(self.messages.ClouddeployProjectsLocationsTargetsPatchRequest(target=target_obj, allowMissing=True, name=target_obj.name, updateMask=TARGET_UPDATE_MASK))

    def Delete(self, name):
        """Deletes a target resource.

    Args:

      name: str, target name.

    Returns:
      The operation message. It could be none if the resource doesn't exist.
    """
        return self._service.Delete(self.messages.ClouddeployProjectsLocationsTargetsDeleteRequest(allowMissing=True, name=name))

    def List(self, location):
        """Lists target resources in a location.

    Args:
      location: str, the full name of the location which owns the targets.

    Returns:
      Returns a list of targets in the given location.
    """
        return self._service.List(self.messages.ClouddeployProjectsLocationsTargetsListRequest(parent=location))
from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.toolresults.v1beta3 import toolresults_v1beta3_messages as messages
def GetSettings(self, request, global_params=None):
    """Gets the Tool Results settings for a project. May return any of the following canonical error codes: - PERMISSION_DENIED - if the user is not authorized to read from project.

      Args:
        request: (ToolresultsProjectsGetSettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ProjectSettings) The response message.
      """
    config = self.GetMethodConfig('GetSettings')
    return self._RunMethod(config, request, global_params=global_params)
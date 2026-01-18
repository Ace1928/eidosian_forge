from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.osconfig.v1 import osconfig_v1_messages as messages
def UpdateProjectFeatureSettings(self, request, global_params=None):
    """UpdateProjectFeatureSettings sets the feature settings for a project.

      Args:
        request: (OsconfigProjectsLocationsGlobalUpdateProjectFeatureSettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ProjectFeatureSettings) The response message.
      """
    config = self.GetMethodConfig('UpdateProjectFeatureSettings')
    return self._RunMethod(config, request, global_params=global_params)
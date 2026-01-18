from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.logging.v2 import logging_v2_messages as messages
def GetCmekSettings(self, request, global_params=None):
    """Gets the Logging CMEK settings for the given resource.Note: CMEK for the Log Router can be configured for Google Cloud projects, folders, organizations, and billing accounts. Once configured for an organization, it applies to all projects and folders in the Google Cloud organization.See Enabling CMEK for Log Router (https://cloud.google.com/logging/docs/routing/managed-encryption) for more information.

      Args:
        request: (LoggingGetCmekSettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CmekSettings) The response message.
      """
    config = self.GetMethodConfig('GetCmekSettings')
    return self._RunMethod(config, request, global_params=global_params)
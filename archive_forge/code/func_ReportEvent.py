from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1beta1 import aiplatform_v1beta1_messages as messages
def ReportEvent(self, request, global_params=None):
    """ReportEvent method for the projects_locations_notebookRuntimes service.

      Args:
        request: (AiplatformProjectsLocationsNotebookRuntimesReportEventRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1beta1ReportRuntimeEventResponse) The response message.
      """
    config = self.GetMethodConfig('ReportEvent')
    return self._RunMethod(config, request, global_params=global_params)
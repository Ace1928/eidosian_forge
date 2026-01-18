from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def ReportStatus(self, request, global_params=None):
    """Reports the latest status for a runtime instance.

      Args:
        request: (ApigeeOrganizationsInstancesReportStatusRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ReportInstanceStatusResponse) The response message.
      """
    config = self.GetMethodConfig('ReportStatus')
    return self._RunMethod(config, request, global_params=global_params)
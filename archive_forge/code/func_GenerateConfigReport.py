from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.servicemanagement.v1 import servicemanagement_v1_messages as messages
def GenerateConfigReport(self, request, global_params=None):
    """Generates and returns a report (errors, warnings and changes from existing configurations) associated with GenerateConfigReportRequest.new_value If GenerateConfigReportRequest.old_value is specified, GenerateConfigReportRequest will contain a single ChangeReport based on the comparison between GenerateConfigReportRequest.new_value and GenerateConfigReportRequest.old_value. If GenerateConfigReportRequest.old_value is not specified, this method will compare GenerateConfigReportRequest.new_value with the last pushed service configuration.

      Args:
        request: (GenerateConfigReportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GenerateConfigReportResponse) The response message.
      """
    config = self.GetMethodConfig('GenerateConfigReport')
    return self._RunMethod(config, request, global_params=global_params)
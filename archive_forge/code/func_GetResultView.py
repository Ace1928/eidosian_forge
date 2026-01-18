from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def GetResultView(self, request, global_params=None):
    """After the query is completed, use this API to view the query result when result size is small.

      Args:
        request: (ApigeeOrganizationsHostSecurityReportsGetResultViewRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1SecurityReportResultView) The response message.
      """
    config = self.GetMethodConfig('GetResultView')
    return self._RunMethod(config, request, global_params=global_params)
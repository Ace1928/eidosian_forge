from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def GetSchemav2(self, request, global_params=None):
    """Gets a list of metrics and dimensions that can be used to create analytics queries and reports. Each schema element contains the name of the field, its associated type, and a flag indicating whether it is a standard or custom field.

      Args:
        request: (ApigeeOrganizationsEnvironmentsAnalyticsAdminGetSchemav2Request) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1Schema) The response message.
      """
    config = self.GetMethodConfig('GetSchemav2')
    return self._RunMethod(config, request, global_params=global_params)
from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def BatchUpdate(self, request, global_params=None):
    """BatchUpdateSecurityIncident updates multiple existing security incidents.

      Args:
        request: (ApigeeOrganizationsEnvironmentsSecurityIncidentsBatchUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1BatchUpdateSecurityIncidentsResponse) The response message.
      """
    config = self.GetMethodConfig('BatchUpdate')
    return self._RunMethod(config, request, global_params=global_params)
from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1alpha2 import healthcare_v1alpha2_messages as messages
def CheckDataAccess(self, request, global_params=None):
    """Checks if a particular data_id of a User data mapping in the specified consent store is consented for the specified use.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresCheckDataAccessRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CheckDataAccessResponse) The response message.
      """
    config = self.GetMethodConfig('CheckDataAccess')
    return self._RunMethod(config, request, global_params=global_params)
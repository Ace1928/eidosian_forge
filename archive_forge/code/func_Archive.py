from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1alpha2 import healthcare_v1alpha2_messages as messages
def Archive(self, request, global_params=None):
    """Archives the specified User data mapping.

      Args:
        request: (HealthcareProjectsLocationsDatasetsConsentStoresUserDataMappingsArchiveRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ArchiveUserDataMappingResponse) The response message.
      """
    config = self.GetMethodConfig('Archive')
    return self._RunMethod(config, request, global_params=global_params)
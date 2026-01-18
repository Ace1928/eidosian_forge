from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1beta1 import healthcare_v1beta1_messages as messages
def ConfigureSearch(self, request, global_params=None):
    """Configure the search parameters for the FHIR store and reindex resources in the FHIR store according to the defined search parameters. The search parameters provided in this request will replace any previous search configuration. The target SearchParameter resources need to exist in the store before calling ConfigureSearch, otherwise an error will occur. This method returns an Operation that can be used to track the progress of the reindexing by calling GetOperation.

      Args:
        request: (HealthcareProjectsLocationsDatasetsFhirStoresConfigureSearchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('ConfigureSearch')
    return self._RunMethod(config, request, global_params=global_params)
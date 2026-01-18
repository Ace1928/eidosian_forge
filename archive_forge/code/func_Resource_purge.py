from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1 import healthcare_v1_messages as messages
def Resource_purge(self, request, global_params=None):
    """Deletes all the historical versions of a resource (excluding the current version) from the FHIR store. To remove all versions of a resource, first delete the current version and then call this method. This is not a FHIR standard operation. For samples that show how to call `Resource-purge`, see [Deleting historical versions of a FHIR resource](https://cloud.google.com/healthcare/docs/how-tos/fhir-resources#deleting_historical_versions_of_a_fhir_resource).

      Args:
        request: (HealthcareProjectsLocationsDatasetsFhirStoresFhirResourcePurgeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
    config = self.GetMethodConfig('Resource_purge')
    return self._RunMethod(config, request, global_params=global_params)
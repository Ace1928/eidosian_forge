from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsFhirStoresRollbackRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsFhirStoresRollbackRequest object.

  Fields:
    name: Required. The name of the FHIR store to rollback, in the format of
      "projects/{project_id}/locations/{location_id}/datasets/{dataset_id}
      /fhirStores/{fhir_store_id}".
    rollbackFhirResourcesRequest: A RollbackFhirResourcesRequest resource to
      be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    rollbackFhirResourcesRequest = _messages.MessageField('RollbackFhirResourcesRequest', 2)
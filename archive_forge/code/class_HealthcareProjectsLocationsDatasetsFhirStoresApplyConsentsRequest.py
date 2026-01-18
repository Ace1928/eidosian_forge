from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsFhirStoresApplyConsentsRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsFhirStoresApplyConsentsRequest
  object.

  Fields:
    applyConsentsRequest: A ApplyConsentsRequest resource to be passed as the
      request body.
    name: Required. The name of the FHIR store to enforce, in the format `proj
      ects/{project_id}/locations/{location_id}/datasets/{dataset_id}/fhirStor
      es/{fhir_store_id}`.
  """
    applyConsentsRequest = _messages.MessageField('ApplyConsentsRequest', 1)
    name = _messages.StringField(2, required=True)
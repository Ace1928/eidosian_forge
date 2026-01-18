from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsFhirStoresDeidentifyRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsFhirStoresDeidentifyRequest object.

  Fields:
    deidentifyFhirStoreRequest: A DeidentifyFhirStoreRequest resource to be
      passed as the request body.
    sourceStore: Required. Source FHIR store resource name. For example, `proj
      ects/{project_id}/locations/{location_id}/datasets/{dataset_id}/fhirStor
      es/{fhir_store_id}`.
  """
    deidentifyFhirStoreRequest = _messages.MessageField('DeidentifyFhirStoreRequest', 1)
    sourceStore = _messages.StringField(2, required=True)
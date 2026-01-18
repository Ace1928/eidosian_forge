from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsFhirStoresFhirConsentEnforcementStatusRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsFhirStoresFhirConsentEnforcementSta
  tusRequest object.

  Fields:
    name: Required. The name of the consent resource to find enforcement
      status, in the format `projects/{project_id}/locations/{location_id}/dat
      asets/{dataset_id}/fhirStores/{fhir_store_id}/fhir/Consent/{consent_id}`
  """
    name = _messages.StringField(1, required=True)
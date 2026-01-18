from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsFhirStoresFhirPatientConsentEnforcementStatusRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsFhirStoresFhirPatientConsentEnforce
  mentStatusRequest object.

  Fields:
    _count: Optional. The maximum number of results on a page. If not
      specified, 100 is used. May not be larger than 1000.
    _page_token: Optional. Used to retrieve the first, previous, next, or last
      page of consent enforcement statuses when using pagination. Value should
      be set to the value of `_page_token` set in next or previous page links'
      URLs. Next and previous page are returned in the response bundle's links
      field, where `link.relation` is "previous" or "next". Omit `_page_token`
      if no previous request has been made.
    name: Required. The name of the patient to find enforcement statuses, in
      the format `projects/{project_id}/locations/{location_id}/datasets/{data
      set_id}/fhirStores/{fhir_store_id}/fhir/Patient/{patient_id}`
  """
    _count = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    _page_token = _messages.StringField(2)
    name = _messages.StringField(3, required=True)
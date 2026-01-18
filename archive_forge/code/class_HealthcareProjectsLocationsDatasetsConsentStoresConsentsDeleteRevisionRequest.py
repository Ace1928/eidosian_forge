from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsConsentStoresConsentsDeleteRevisionRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsConsentStoresConsentsDeleteRevision
  Request object.

  Fields:
    name: Required. The resource name of the Consent revision to delete, of
      the form `projects/{project_id}/locations/{location_id}/datasets/{datase
      t_id}/consentStores/{consent_store_id}/consents/{consent_id}@{revision_i
      d}`. An INVALID_ARGUMENT error occurs if `revision_id` is not specified
      in the name.
  """
    name = _messages.StringField(1, required=True)
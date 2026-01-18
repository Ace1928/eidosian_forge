from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsConsentStoresConsentsActivateRequest(_messages.Message):
    """A
  HealthcareProjectsLocationsDatasetsConsentStoresConsentsActivateRequest
  object.

  Fields:
    activateConsentRequest: A ActivateConsentRequest resource to be passed as
      the request body.
    name: Required. The resource name of the Consent to activate, of the form
      `projects/{project_id}/locations/{location_id}/datasets/{dataset_id}/con
      sentStores/{consent_store_id}/consents/{consent_id}`. An
      INVALID_ARGUMENT error occurs if `revision_id` is specified in the name.
  """
    activateConsentRequest = _messages.MessageField('ActivateConsentRequest', 1)
    name = _messages.StringField(2, required=True)
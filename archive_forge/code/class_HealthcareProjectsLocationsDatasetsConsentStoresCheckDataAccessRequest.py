from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsConsentStoresCheckDataAccessRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsConsentStoresCheckDataAccessRequest
  object.

  Fields:
    checkDataAccessRequest: A CheckDataAccessRequest resource to be passed as
      the request body.
    consentStore: Required. Name of the consent store where the requested
      data_id is stored, of the form `projects/{project_id}/locations/{locatio
      n_id}/datasets/{dataset_id}/consentStores/{consent_store_id}`.
  """
    checkDataAccessRequest = _messages.MessageField('CheckDataAccessRequest', 1)
    consentStore = _messages.StringField(2, required=True)
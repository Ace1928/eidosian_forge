from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsConsentStoresConsentsCreateRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsConsentStoresConsentsCreateRequest
  object.

  Fields:
    consent: A Consent resource to be passed as the request body.
    parent: Required. Name of the consent store.
  """
    consent = _messages.MessageField('Consent', 1)
    parent = _messages.StringField(2, required=True)
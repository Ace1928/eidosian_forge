from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsConsentStoresConsentArtifactsCreateRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsConsentStoresConsentArtifactsCreate
  Request object.

  Fields:
    consentArtifact: A ConsentArtifact resource to be passed as the request
      body.
    parent: Required. The name of the consent store this Consent artifact
      belongs to.
  """
    consentArtifact = _messages.MessageField('ConsentArtifact', 1)
    parent = _messages.StringField(2, required=True)
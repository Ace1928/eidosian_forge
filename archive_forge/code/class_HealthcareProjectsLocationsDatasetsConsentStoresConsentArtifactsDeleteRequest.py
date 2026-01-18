from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsConsentStoresConsentArtifactsDeleteRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsConsentStoresConsentArtifactsDelete
  Request object.

  Fields:
    name: Required. The resource name of the Consent artifact to delete. To
      preserve referential integrity, Consent artifacts referenced by the
      latest revision of a Consent cannot be deleted.
  """
    name = _messages.StringField(1, required=True)
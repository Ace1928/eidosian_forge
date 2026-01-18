from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsConsentStoresAttributeDefinitionsDeleteRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsConsentStoresAttributeDefinitionsDe
  leteRequest object.

  Fields:
    name: Required. The resource name of the Attribute definition to delete.
      To preserve referential integrity, Attribute definitions referenced by a
      User data mapping or the latest revision of a Consent cannot be deleted.
  """
    name = _messages.StringField(1, required=True)
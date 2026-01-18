from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsConsentStoresAttributeDefinitionsGetRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsConsentStoresAttributeDefinitionsGe
  tRequest object.

  Fields:
    name: Required. The resource name of the Attribute definition to get.
  """
    name = _messages.StringField(1, required=True)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsConsentStoresUserDataMappingsDeleteRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsConsentStoresUserDataMappingsDelete
  Request object.

  Fields:
    name: Required. The resource name of the User data mapping to delete.
  """
    name = _messages.StringField(1, required=True)
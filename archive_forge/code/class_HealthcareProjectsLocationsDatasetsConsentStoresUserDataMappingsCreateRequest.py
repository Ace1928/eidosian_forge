from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsConsentStoresUserDataMappingsCreateRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsConsentStoresUserDataMappingsCreate
  Request object.

  Fields:
    parent: Required. Name of the consent store.
    userDataMapping: A UserDataMapping resource to be passed as the request
      body.
  """
    parent = _messages.StringField(1, required=True)
    userDataMapping = _messages.MessageField('UserDataMapping', 2)
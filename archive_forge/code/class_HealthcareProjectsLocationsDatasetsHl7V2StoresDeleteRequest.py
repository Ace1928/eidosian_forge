from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsHl7V2StoresDeleteRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsHl7V2StoresDeleteRequest object.

  Fields:
    name: Required. The resource name of the HL7v2 store to delete.
  """
    name = _messages.StringField(1, required=True)
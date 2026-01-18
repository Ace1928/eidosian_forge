from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsHl7V2StoresGetRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsHl7V2StoresGetRequest object.

  Fields:
    name: Required. The resource name of the HL7v2 store to get.
  """
    name = _messages.StringField(1, required=True)
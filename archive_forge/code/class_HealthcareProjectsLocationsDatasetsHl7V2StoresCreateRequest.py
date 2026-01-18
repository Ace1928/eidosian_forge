from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsHl7V2StoresCreateRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsHl7V2StoresCreateRequest object.

  Fields:
    hl7V2Store: A Hl7V2Store resource to be passed as the request body.
    hl7V2StoreId: Required. The ID of the HL7v2 store that is being created.
      The string must match the following regex: `[\\p{L}\\p{N}_\\-\\.]{1,256}`.
    parent: Required. The name of the dataset this HL7v2 store belongs to.
  """
    hl7V2Store = _messages.MessageField('Hl7V2Store', 1)
    hl7V2StoreId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DnsManagedZonesPatchRequest(_messages.Message):
    """A DnsManagedZonesPatchRequest object.

  Fields:
    clientOperationId: For mutating operation requests only. An optional
      identifier specified by the client. Must be unique for operation
      resources in the Operations collection.
    managedZone: Identifies the managed zone addressed by this request. Can be
      the managed zone name or ID.
    managedZoneResource: A ManagedZone resource to be passed as the request
      body.
    project: Identifies the project addressed by this request.
  """
    clientOperationId = _messages.StringField(1)
    managedZone = _messages.StringField(2, required=True)
    managedZoneResource = _messages.MessageField('ManagedZone', 3)
    project = _messages.StringField(4, required=True)
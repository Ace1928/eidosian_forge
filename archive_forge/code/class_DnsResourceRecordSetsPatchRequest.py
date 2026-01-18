from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DnsResourceRecordSetsPatchRequest(_messages.Message):
    """A DnsResourceRecordSetsPatchRequest object.

  Fields:
    clientOperationId: For mutating operation requests only. An optional
      identifier specified by the client. Must be unique for operation
      resources in the Operations collection.
    managedZone: Identifies the managed zone addressed by this request. Can be
      the managed zone name or ID.
    name: Fully qualified domain name.
    project: Identifies the project addressed by this request.
    resourceRecordSet: A ResourceRecordSet resource to be passed as the
      request body.
    type: RRSet type.
  """
    clientOperationId = _messages.StringField(1)
    managedZone = _messages.StringField(2, required=True)
    name = _messages.StringField(3, required=True)
    project = _messages.StringField(4, required=True)
    resourceRecordSet = _messages.MessageField('ResourceRecordSet', 5)
    type = _messages.StringField(6, required=True)
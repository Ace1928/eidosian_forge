from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DnsResponsePoliciesCreateRequest(_messages.Message):
    """A DnsResponsePoliciesCreateRequest object.

  Fields:
    clientOperationId: For mutating operation requests only. An optional
      identifier specified by the client. Must be unique for operation
      resources in the Operations collection.
    project: Identifies the project addressed by this request.
    responsePolicy: A ResponsePolicy resource to be passed as the request
      body.
  """
    clientOperationId = _messages.StringField(1)
    project = _messages.StringField(2, required=True)
    responsePolicy = _messages.MessageField('ResponsePolicy', 3)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DnsPoliciesPatchRequest(_messages.Message):
    """A DnsPoliciesPatchRequest object.

  Fields:
    clientOperationId: For mutating operation requests only. An optional
      identifier specified by the client. Must be unique for operation
      resources in the Operations collection.
    policy: User given friendly name of the policy addressed by this request.
    policyResource: A Policy resource to be passed as the request body.
    project: Identifies the project addressed by this request.
  """
    clientOperationId = _messages.StringField(1)
    policy = _messages.StringField(2, required=True)
    policyResource = _messages.MessageField('Policy', 3)
    project = _messages.StringField(4, required=True)
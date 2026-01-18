from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DnsResponsePolicyRulesUpdateRequest(_messages.Message):
    """A DnsResponsePolicyRulesUpdateRequest object.

  Fields:
    clientOperationId: For mutating operation requests only. An optional
      identifier specified by the client. Must be unique for operation
      resources in the Operations collection.
    project: Identifies the project addressed by this request.
    responsePolicy: User assigned name of the Response Policy containing the
      Response Policy Rule.
    responsePolicyRule: User assigned name of the Response Policy Rule
      addressed by this request.
    responsePolicyRuleResource: A ResponsePolicyRule resource to be passed as
      the request body.
  """
    clientOperationId = _messages.StringField(1)
    project = _messages.StringField(2, required=True)
    responsePolicy = _messages.StringField(3, required=True)
    responsePolicyRule = _messages.StringField(4, required=True)
    responsePolicyRuleResource = _messages.MessageField('ResponsePolicyRule', 5)
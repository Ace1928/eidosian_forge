from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ResponsePolicyRulesUpdateResponse(_messages.Message):
    """A ResponsePolicyRulesUpdateResponse object.

  Fields:
    header: A ResponseHeader attribute.
    responsePolicyRule: A ResponsePolicyRule attribute.
  """
    header = _messages.MessageField('ResponseHeader', 1)
    responsePolicyRule = _messages.MessageField('ResponsePolicyRule', 2)
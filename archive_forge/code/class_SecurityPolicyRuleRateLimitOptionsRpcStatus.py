from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityPolicyRuleRateLimitOptionsRpcStatus(_messages.Message):
    """Simplified google.rpc.Status type (omitting details).

  Fields:
    code: The status code, which should be an enum value of google.rpc.Code.
    message: A developer-facing error message, which should be in English.
  """
    code = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    message = _messages.StringField(2)
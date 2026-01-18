from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityPolicyRuleHttpHeaderActionHttpHeaderOption(_messages.Message):
    """A SecurityPolicyRuleHttpHeaderActionHttpHeaderOption object.

  Fields:
    headerName: The name of the header to set.
    headerValue: The value to set the named header to.
  """
    headerName = _messages.StringField(1)
    headerValue = _messages.StringField(2)
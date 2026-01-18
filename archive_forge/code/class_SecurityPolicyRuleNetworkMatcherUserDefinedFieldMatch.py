from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityPolicyRuleNetworkMatcherUserDefinedFieldMatch(_messages.Message):
    """A SecurityPolicyRuleNetworkMatcherUserDefinedFieldMatch object.

  Fields:
    name: Name of the user-defined field, as given in the definition.
    values: Matching values of the field. Each element can be a 32-bit
      unsigned decimal or hexadecimal (starting with "0x") number (e.g. "64")
      or range (e.g. "0x400-0x7ff").
  """
    name = _messages.StringField(1)
    values = _messages.StringField(2, repeated=True)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV2PolicyRule(_messages.Message):
    """A single rule in a `Policy`.

  Fields:
    denyRule: A rule for a deny policy.
    description: A user-specified description of the rule. This value can be
      up to 256 characters.
  """
    denyRule = _messages.MessageField('GoogleIamV2DenyRule', 1)
    description = _messages.StringField(2)
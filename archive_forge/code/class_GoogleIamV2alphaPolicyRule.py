from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV2alphaPolicyRule(_messages.Message):
    """A single rule in a `Policy`.

  Fields:
    denyRule: Specification of a Deny `Policy`.
    description: A user-specified opaque description of the rule. Must be less
      than or equal to 256 characters.
  """
    denyRule = _messages.MessageField('GoogleIamV2alphaDenyRule', 1)
    description = _messages.StringField(2)
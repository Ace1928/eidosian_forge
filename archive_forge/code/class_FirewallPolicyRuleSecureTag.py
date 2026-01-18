from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirewallPolicyRuleSecureTag(_messages.Message):
    """A FirewallPolicyRuleSecureTag object.

  Enums:
    StateValueValuesEnum: [Output Only] State of the secure tag, either
      `EFFECTIVE` or `INEFFECTIVE`. A secure tag is `INEFFECTIVE` when it is
      deleted or its network is deleted.

  Fields:
    name: Name of the secure tag, created with TagManager's TagValue API.
    state: [Output Only] State of the secure tag, either `EFFECTIVE` or
      `INEFFECTIVE`. A secure tag is `INEFFECTIVE` when it is deleted or its
      network is deleted.
  """

    class StateValueValuesEnum(_messages.Enum):
        """[Output Only] State of the secure tag, either `EFFECTIVE` or
    `INEFFECTIVE`. A secure tag is `INEFFECTIVE` when it is deleted or its
    network is deleted.

    Values:
      EFFECTIVE: <no description>
      INEFFECTIVE: <no description>
    """
        EFFECTIVE = 0
        INEFFECTIVE = 1
    name = _messages.StringField(1)
    state = _messages.EnumField('StateValueValuesEnum', 2)
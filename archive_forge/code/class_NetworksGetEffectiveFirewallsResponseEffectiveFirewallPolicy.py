from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksGetEffectiveFirewallsResponseEffectiveFirewallPolicy(_messages.Message):
    """A NetworksGetEffectiveFirewallsResponseEffectiveFirewallPolicy object.

  Enums:
    TypeValueValuesEnum: [Output Only] The type of the firewall policy.

  Fields:
    displayName: [Output Only] Deprecated, please use short name instead. The
      display name of the firewall policy.
    name: [Output Only] The name of the firewall policy.
    rules: The rules that apply to the network.
    shortName: [Output Only] The short name of the firewall policy.
    type: [Output Only] The type of the firewall policy.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """[Output Only] The type of the firewall policy.

    Values:
      HIERARCHY: <no description>
      NETWORK: <no description>
      UNSPECIFIED: <no description>
    """
        HIERARCHY = 0
        NETWORK = 1
        UNSPECIFIED = 2
    displayName = _messages.StringField(1)
    name = _messages.StringField(2)
    rules = _messages.MessageField('FirewallPolicyRule', 3, repeated=True)
    shortName = _messages.StringField(4)
    type = _messages.EnumField('TypeValueValuesEnum', 5)
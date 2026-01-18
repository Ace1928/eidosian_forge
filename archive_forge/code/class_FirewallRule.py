from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirewallRule(_messages.Message):
    """A single firewall rule that is evaluated against incoming traffic and
  provides an action to take on matched requests.

  Enums:
    ActionValueValuesEnum: The action to take on matched requests.

  Fields:
    action: The action to take on matched requests.
    description: An optional string description of this rule. This field has a
      maximum length of 400 characters.
    priority: A positive integer between 1, Int32.MaxValue-1 that defines the
      order of rule evaluation. Rules with the lowest priority are evaluated
      first.A default rule at priority Int32.MaxValue matches all IPv4 and
      IPv6 traffic when no previous rule matches. Only the action of this rule
      can be modified by the user.
    sourceRange: IP address or range, defined using CIDR notation, of requests
      that this rule applies to. You can use the wildcard character "*" to
      match all IPs equivalent to "0/0" and "::/0" together. Examples:
      192.168.1.1 or 192.168.0.0/16 or 2001:db8::/32 or
      2001:0db8:0000:0042:0000:8a2e:0370:7334. Truncation will be silently
      performed on addresses which are not properly truncated. For example,
      1.2.3.4/24 is accepted as the same address as 1.2.3.0/24. Similarly, for
      IPv6, 2001:db8::1/32 is accepted as the same address as 2001:db8::/32.
  """

    class ActionValueValuesEnum(_messages.Enum):
        """The action to take on matched requests.

    Values:
      UNSPECIFIED_ACTION: <no description>
      ALLOW: Matching requests are allowed.
      DENY: Matching requests are denied.
    """
        UNSPECIFIED_ACTION = 0
        ALLOW = 1
        DENY = 2
    action = _messages.EnumField('ActionValueValuesEnum', 1)
    description = _messages.StringField(2)
    priority = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    sourceRange = _messages.StringField(4)
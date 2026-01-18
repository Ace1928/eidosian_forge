from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityPolicyRuleMatcherConfig(_messages.Message):
    """A SecurityPolicyRuleMatcherConfig object.

  Fields:
    destIpRanges: CIDR IP address range. This field may only be specified when
      versioned_expr is set to FIREWALL.
    layer4Configs: Pairs of IP protocols and ports that the rule should match.
      This field may only be specified when versioned_expr is set to FIREWALL.
    srcIpRanges: CIDR IP address range. Maximum number of src_ip_ranges
      allowed is 10.
  """
    destIpRanges = _messages.StringField(1, repeated=True)
    layer4Configs = _messages.MessageField('SecurityPolicyRuleMatcherConfigLayer4Config', 2, repeated=True)
    srcIpRanges = _messages.StringField(3, repeated=True)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GatewaySecurityPolicyRule(_messages.Message):
    """The GatewaySecurityPolicyRule resource is in a nested collection within
  a GatewaySecurityPolicy and represents a traffic matching condition and
  associated action to perform.

  Enums:
    BasicProfileValueValuesEnum: Required. Profile which tells what the
      primitive action should be.

  Fields:
    applicationMatcher: Optional. CEL expression for matching on
      L7/application level criteria.
    basicProfile: Required. Profile which tells what the primitive action
      should be.
    createTime: Output only. Time when the rule was created.
    description: Optional. Free-text description of the resource.
    enabled: Required. Whether the rule is enforced.
    name: Required. Immutable. Name of the resource. ame is the full resource
      name so projects/{project}/locations/{location}/gatewaySecurityPolicies/
      {gateway_security_policy}/rules/{rule} rule should match the pattern:
      (^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$).
    priority: Required. Priority of the rule. Lower number corresponds to
      higher precedence.
    sessionMatcher: Required. CEL expression for matching on session criteria.
    tlsInspectionEnabled: Optional. Flag to enable TLS inspection of traffic
      matching on , can only be true if the parent GatewaySecurityPolicy
      references a TLSInspectionConfig.
    updateTime: Output only. Time when the rule was updated.
  """

    class BasicProfileValueValuesEnum(_messages.Enum):
        """Required. Profile which tells what the primitive action should be.

    Values:
      BASIC_PROFILE_UNSPECIFIED: If there is not a mentioned action for the
        target.
      ALLOW: Allow the matched traffic.
      DENY: Deny the matched traffic.
    """
        BASIC_PROFILE_UNSPECIFIED = 0
        ALLOW = 1
        DENY = 2
    applicationMatcher = _messages.StringField(1)
    basicProfile = _messages.EnumField('BasicProfileValueValuesEnum', 2)
    createTime = _messages.StringField(3)
    description = _messages.StringField(4)
    enabled = _messages.BooleanField(5)
    name = _messages.StringField(6)
    priority = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    sessionMatcher = _messages.StringField(8)
    tlsInspectionEnabled = _messages.BooleanField(9)
    updateTime = _messages.StringField(10)
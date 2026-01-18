from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirewallPolicyRule(_messages.Message):
    """Represents a rule that describes one or more match conditions along with
  the action to be taken when traffic matches this condition (allow or deny).

  Enums:
    DirectionValueValuesEnum: The direction in which this rule applies.

  Fields:
    action: The Action to perform when the client connection triggers the
      rule. Valid actions are "allow", "deny" and "goto_next".
    description: An optional description for this resource.
    direction: The direction in which this rule applies.
    disabled: Denotes whether the firewall policy rule is disabled. When set
      to true, the firewall policy rule is not enforced and traffic behaves as
      if it did not exist. If this is unspecified, the firewall policy rule
      will be enabled.
    enableLogging: Denotes whether to enable logging for a particular rule. If
      logging is enabled, logs will be exported to the configured export
      destination in Stackdriver. Logs may be exported to BigQuery or Pub/Sub.
      Note: you cannot enable logging on "goto_next" rules.
    kind: [Output only] Type of the resource. Always
      compute#firewallPolicyRule for firewall policy rules
    match: A match condition that incoming traffic is evaluated against. If it
      evaluates to true, the corresponding 'action' is enforced.
    priority: An integer indicating the priority of a rule in the list. The
      priority must be a positive value between 0 and 2147483647. Rules are
      evaluated from highest to lowest priority where 0 is the highest
      priority and 2147483647 is the lowest prority.
    ruleName: An optional name for the rule. This field is not a unique
      identifier and can be updated.
    ruleTupleCount: [Output Only] Calculation of the complexity of a single
      firewall policy rule.
    securityProfileGroup: A fully-qualified URL of a SecurityProfile resource
      instance. Example: https://networksecurity.googleapis.com/v1/projects/{p
      roject}/locations/{location}/securityProfileGroups/my-security-profile-
      group Must be specified if action = 'apply_security_profile_group' and
      cannot be specified for other actions.
    targetResources: A list of network resource URLs to which this rule
      applies. This field allows you to control which network's VMs get this
      rule. If this field is left blank, all VMs within the organization will
      receive the rule.
    targetSecureTags: A list of secure tags that controls which instances the
      firewall rule applies to. If targetSecureTag are specified, then the
      firewall rule applies only to instances in the VPC network that have one
      of those EFFECTIVE secure tags, if all the target_secure_tag are in
      INEFFECTIVE state, then this rule will be ignored. targetSecureTag may
      not be set at the same time as targetServiceAccounts. If neither
      targetServiceAccounts nor targetSecureTag are specified, the firewall
      rule applies to all instances on the specified network. Maximum number
      of target label tags allowed is 256.
    targetServiceAccounts: A list of service accounts indicating the sets of
      instances that are applied with this rule.
    tlsInspect: Boolean flag indicating if the traffic should be TLS
      decrypted. Can be set only if action = 'apply_security_profile_group'
      and cannot be set for other actions.
  """

    class DirectionValueValuesEnum(_messages.Enum):
        """The direction in which this rule applies.

    Values:
      EGRESS: <no description>
      INGRESS: <no description>
    """
        EGRESS = 0
        INGRESS = 1
    action = _messages.StringField(1)
    description = _messages.StringField(2)
    direction = _messages.EnumField('DirectionValueValuesEnum', 3)
    disabled = _messages.BooleanField(4)
    enableLogging = _messages.BooleanField(5)
    kind = _messages.StringField(6, default='compute#firewallPolicyRule')
    match = _messages.MessageField('FirewallPolicyRuleMatcher', 7)
    priority = _messages.IntegerField(8, variant=_messages.Variant.INT32)
    ruleName = _messages.StringField(9)
    ruleTupleCount = _messages.IntegerField(10, variant=_messages.Variant.INT32)
    securityProfileGroup = _messages.StringField(11)
    targetResources = _messages.StringField(12, repeated=True)
    targetSecureTags = _messages.MessageField('FirewallPolicyRuleSecureTag', 13, repeated=True)
    targetServiceAccounts = _messages.StringField(14, repeated=True)
    tlsInspect = _messages.BooleanField(15)
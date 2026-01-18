from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityPolicyRuleMatcher(_messages.Message):
    """Represents a match condition that incoming traffic is evaluated against.
  Exactly one field must be specified.

  Enums:
    VersionedExprValueValuesEnum: Preconfigured versioned expression. If this
      field is specified, config must also be specified. Available
      preconfigured expressions along with their requirements are: SRC_IPS_V1
      - must specify the corresponding src_ip_range field in config.

  Fields:
    config: The configuration options available when specifying
      versioned_expr. This field must be specified if versioned_expr is
      specified and cannot be specified if versioned_expr is not specified.
    expr: User defined CEVAL expression. A CEVAL expression is used to specify
      match criteria such as origin.ip, source.region_code and contents in the
      request header. Expressions containing `evaluateThreatIntelligence`
      require Cloud Armor Managed Protection Plus tier and are not supported
      in Edge Policies nor in Regional Policies. Expressions containing
      `evaluatePreconfiguredExpr('sourceiplist-*')` require Cloud Armor
      Managed Protection Plus tier and are only supported in Global Security
      Policies.
    exprOptions: The configuration options available when specifying a user
      defined CEVAL expression (i.e., 'expr').
    versionedExpr: Preconfigured versioned expression. If this field is
      specified, config must also be specified. Available preconfigured
      expressions along with their requirements are: SRC_IPS_V1 - must specify
      the corresponding src_ip_range field in config.
  """

    class VersionedExprValueValuesEnum(_messages.Enum):
        """Preconfigured versioned expression. If this field is specified, config
    must also be specified. Available preconfigured expressions along with
    their requirements are: SRC_IPS_V1 - must specify the corresponding
    src_ip_range field in config.

    Values:
      FIREWALL: <no description>
      SRC_IPS_V1: Matches the source IP address of a request to the IP ranges
        supplied in config.
    """
        FIREWALL = 0
        SRC_IPS_V1 = 1
    config = _messages.MessageField('SecurityPolicyRuleMatcherConfig', 1)
    expr = _messages.MessageField('Expr', 2)
    exprOptions = _messages.MessageField('SecurityPolicyRuleMatcherExprOptions', 3)
    versionedExpr = _messages.EnumField('VersionedExprValueValuesEnum', 4)
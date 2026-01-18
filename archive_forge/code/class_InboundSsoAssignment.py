from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InboundSsoAssignment(_messages.Message):
    """Targets with "set" SSO assignments and their respective assignments.

  Enums:
    SsoModeValueValuesEnum: Inbound SSO behavior.

  Fields:
    customer: Immutable. The customer. For example: `customers/C0123abc`.
    name: Output only. [Resource
      name](https://cloud.google.com/apis/design/resource_names) of the
      Inbound SSO Assignment.
    rank: Must be zero (which is the default value so it can be omitted) for
      assignments with `target_org_unit` set and must be greater-than-or-
      equal-to one for assignments with `target_group` set.
    samlSsoInfo: SAML SSO details. Must be set if and only if `sso_mode` is
      set to `SAML_SSO`.
    signInBehavior: Assertions about users assigned to an IdP will always be
      accepted from that IdP. This controls whether/when Google should
      redirect a user to the IdP. Unset (defaults) is the recommended
      configuration.
    ssoMode: Inbound SSO behavior.
    targetGroup: Immutable. Must be of the form `groups/{group}`.
    targetOrgUnit: Immutable. Must be of the form `orgUnits/{org_unit}`.
  """

    class SsoModeValueValuesEnum(_messages.Enum):
        """Inbound SSO behavior.

    Values:
      SSO_MODE_UNSPECIFIED: Not allowed.
      SSO_OFF: Disable SSO for the targeted users.
      SAML_SSO: Use an external SAML Identity Provider for SSO for the
        targeted users.
      DOMAIN_WIDE_SAML_IF_ENABLED: Use the domain-wide SAML Identity Provider
        for the targeted users if one is configured; otherwise, this is
        equivalent to `SSO_OFF`. Note that this will also be equivalent to
        `SSO_OFF` if/when support for domain-wide SAML is removed. Google may
        disallow this mode at that point and existing assignments with this
        mode may be automatically changed to `SSO_OFF`.
    """
        SSO_MODE_UNSPECIFIED = 0
        SSO_OFF = 1
        SAML_SSO = 2
        DOMAIN_WIDE_SAML_IF_ENABLED = 3
    customer = _messages.StringField(1)
    name = _messages.StringField(2)
    rank = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    samlSsoInfo = _messages.MessageField('SamlSsoInfo', 4)
    signInBehavior = _messages.MessageField('SignInBehavior', 5)
    ssoMode = _messages.EnumField('SsoModeValueValuesEnum', 6)
    targetGroup = _messages.StringField(7)
    targetOrgUnit = _messages.StringField(8)
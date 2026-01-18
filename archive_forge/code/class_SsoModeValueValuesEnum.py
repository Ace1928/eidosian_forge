from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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
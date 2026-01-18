from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InboundSamlSsoProfile(_messages.Message):
    """A [SAML 2.0](https://www.oasis-open.org/standards#samlv2.0) federation
  between a Google enterprise customer and a SAML identity provider.

  Fields:
    customer: Immutable. The customer. For example: `customers/C0123abc`.
    displayName: Human-readable name of the SAML SSO profile.
    idpConfig: SAML identity provider configuration.
    name: Output only. [Resource
      name](https://cloud.google.com/apis/design/resource_names) of the SAML
      SSO profile.
    spConfig: SAML service provider configuration for this SAML SSO profile.
      These are the service provider details provided by Google that should be
      configured on the corresponding identity provider.
  """
    customer = _messages.StringField(1)
    displayName = _messages.StringField(2)
    idpConfig = _messages.MessageField('SamlIdpConfig', 3)
    name = _messages.StringField(4)
    spConfig = _messages.MessageField('SamlSpConfig', 5)
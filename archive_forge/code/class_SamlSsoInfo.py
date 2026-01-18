from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SamlSsoInfo(_messages.Message):
    """Details that are applicable when `sso_mode` == `SAML_SSO`.

  Fields:
    inboundSamlSsoProfile: Required. Name of the `InboundSamlSsoProfile` to
      use. Must be of the form
      `inboundSamlSsoProfiles/{inbound_saml_sso_profile}`.
  """
    inboundSamlSsoProfile = _messages.StringField(1)
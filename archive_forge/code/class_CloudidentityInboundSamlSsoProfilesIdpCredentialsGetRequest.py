from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityInboundSamlSsoProfilesIdpCredentialsGetRequest(_messages.Message):
    """A CloudidentityInboundSamlSsoProfilesIdpCredentialsGetRequest object.

  Fields:
    name: Required. The [resource
      name](https://cloud.google.com/apis/design/resource_names) of the
      IdpCredential to retrieve. Format: `inboundSamlSsoProfiles/{sso_profile_
      id}/idpCredentials/{idp_credential_id}`
  """
    name = _messages.StringField(1, required=True)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamAdminV1WorkforcePoolProviderExtraAttributesOAuth2Client(_messages.Message):
    """Represents the OAuth 2.0 client credential configuration for retrieving
  additional user attributes that are not present in the initial
  authentication credentials from the identity provider, e.g. groups. See
  https://datatracker.ietf.org/doc/html/rfc6749#section-4.4 for more details
  on client credentials grant flow.

  Enums:
    AttributesTypeValueValuesEnum: Required. Represents the IdP and type of
      claims that should be fetched.

  Fields:
    attributesType: Required. Represents the IdP and type of claims that
      should be fetched.
    clientId: Required. The OAuth 2.0 client ID for retrieving extra
      attributes from the identity provider. Required to get the Access Token
      using client credentials grant flow.
    clientSecret: Required. The OAuth 2.0 client secret for retrieving extra
      attributes from the identity provider. Required to get the Access Token
      using client credentials grant flow.
    issuerUri: Required. The OIDC identity provider's issuer URI. Must be a
      valid URI using the 'https' scheme. Required to get the OIDC discovery
      document.
    queryParameters: Optional. Represents the parameters to control which
      claims are fetched from an IdP.
  """

    class AttributesTypeValueValuesEnum(_messages.Enum):
        """Required. Represents the IdP and type of claims that should be
    fetched.

    Values:
      ATTRIBUTES_TYPE_UNSPECIFIED: No AttributesType specified.
      AZURE_AD_GROUPS_MAIL: Used to get the user's group claims from the Azure
        AD identity provider using configuration provided in
        ExtraAttributesOAuth2Client and `mail` property of the
        `microsoft.graph.group` object is used for claim mapping. See
        https://learn.microsoft.com/en-
        us/graph/api/resources/group?view=graph-rest-1.0#properties for more
        details on `microsoft.graph.group` properties. The attributes obtained
        from idntity provider are mapped to `assertion.groups`.
    """
        ATTRIBUTES_TYPE_UNSPECIFIED = 0
        AZURE_AD_GROUPS_MAIL = 1
    attributesType = _messages.EnumField('AttributesTypeValueValuesEnum', 1)
    clientId = _messages.StringField(2)
    clientSecret = _messages.MessageField('GoogleIamAdminV1WorkforcePoolProviderOidcClientSecret', 3)
    issuerUri = _messages.StringField(4)
    queryParameters = _messages.MessageField('GoogleIamAdminV1WorkforcePoolProviderExtraAttributesOAuth2ClientQueryParameters', 5)
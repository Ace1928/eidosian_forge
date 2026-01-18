from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamAdminV1WorkforcePoolProviderOidcWebSsoConfig(_messages.Message):
    """Configuration for web single sign-on for the OIDC provider.

  Enums:
    AssertionClaimsBehaviorValueValuesEnum: Required. The behavior for how
      OIDC Claims are included in the `assertion` object used for attribute
      mapping and attribute condition.
    ResponseTypeValueValuesEnum: Required. The Response Type to request for in
      the OIDC Authorization Request for web sign-in. The `CODE` Response Type
      is recommended to avoid the Implicit Flow, for security reasons.

  Fields:
    additionalScopes: Additional scopes to request for in the OIDC
      authentication request on top of scopes requested by default. By
      default, the `openid`, `profile` and `email` scopes that are supported
      by the identity provider are requested. Each additional scope may be at
      most 256 characters. A maximum of 10 additional scopes may be
      configured.
    assertionClaimsBehavior: Required. The behavior for how OIDC Claims are
      included in the `assertion` object used for attribute mapping and
      attribute condition.
    responseType: Required. The Response Type to request for in the OIDC
      Authorization Request for web sign-in. The `CODE` Response Type is
      recommended to avoid the Implicit Flow, for security reasons.
  """

    class AssertionClaimsBehaviorValueValuesEnum(_messages.Enum):
        """Required. The behavior for how OIDC Claims are included in the
    `assertion` object used for attribute mapping and attribute condition.

    Values:
      ASSERTION_CLAIMS_BEHAVIOR_UNSPECIFIED: No assertion claims behavior
        specified.
      MERGE_USER_INFO_OVER_ID_TOKEN_CLAIMS: Merge the UserInfo Endpoint Claims
        with ID Token Claims, preferring UserInfo Claim Values for the same
        Claim Name. This option is available only for the Authorization Code
        Flow.
      ONLY_ID_TOKEN_CLAIMS: Only include ID Token Claims.
    """
        ASSERTION_CLAIMS_BEHAVIOR_UNSPECIFIED = 0
        MERGE_USER_INFO_OVER_ID_TOKEN_CLAIMS = 1
        ONLY_ID_TOKEN_CLAIMS = 2

    class ResponseTypeValueValuesEnum(_messages.Enum):
        """Required. The Response Type to request for in the OIDC Authorization
    Request for web sign-in. The `CODE` Response Type is recommended to avoid
    the Implicit Flow, for security reasons.

    Values:
      RESPONSE_TYPE_UNSPECIFIED: No Response Type specified.
      CODE: The `response_type=code` selection uses the Authorization Code
        Flow for web sign-in. Requires a configured client secret.
      ID_TOKEN: The `response_type=id_token` selection uses the Implicit Flow
        for web sign-in.
    """
        RESPONSE_TYPE_UNSPECIFIED = 0
        CODE = 1
        ID_TOKEN = 2
    additionalScopes = _messages.StringField(1, repeated=True)
    assertionClaimsBehavior = _messages.EnumField('AssertionClaimsBehaviorValueValuesEnum', 2)
    responseType = _messages.EnumField('ResponseTypeValueValuesEnum', 3)
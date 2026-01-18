from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuthConfigTemplate(_messages.Message):
    """AuthConfigTemplate defines required field over an authentication type.

  Enums:
    AuthTypeValueValuesEnum: The type of authentication configured.

  Fields:
    authKey: Identifier key for auth config
    authType: The type of authentication configured.
    configVariableTemplates: Config variables to describe an `AuthConfig` for
      a `Connection`.
    description: Connector specific description for an authentication
      template.
    displayName: Display name for authentication template.
  """

    class AuthTypeValueValuesEnum(_messages.Enum):
        """The type of authentication configured.

    Values:
      AUTH_TYPE_UNSPECIFIED: Authentication type not specified.
      USER_PASSWORD: Username and Password Authentication.
      OAUTH2_JWT_BEARER: JSON Web Token (JWT) Profile for Oauth 2.0
        Authorization Grant based authentication
      OAUTH2_CLIENT_CREDENTIALS: Oauth 2.0 Client Credentials Grant
        Authentication
      SSH_PUBLIC_KEY: SSH Public Key Authentication
      OAUTH2_AUTH_CODE_FLOW: Oauth 2.0 Authorization Code Flow
    """
        AUTH_TYPE_UNSPECIFIED = 0
        USER_PASSWORD = 1
        OAUTH2_JWT_BEARER = 2
        OAUTH2_CLIENT_CREDENTIALS = 3
        SSH_PUBLIC_KEY = 4
        OAUTH2_AUTH_CODE_FLOW = 5
    authKey = _messages.StringField(1)
    authType = _messages.EnumField('AuthTypeValueValuesEnum', 2)
    configVariableTemplates = _messages.MessageField('ConfigVariableTemplate', 3, repeated=True)
    description = _messages.StringField(4)
    displayName = _messages.StringField(5)
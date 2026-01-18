from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GenerateAccessTokenRequest(_messages.Message):
    """A GenerateAccessTokenRequest object.

  Fields:
    delegates: The sequence of service accounts in a delegation chain. Each
      service account must be granted the
      `roles/iam.serviceAccountTokenCreator` role on its next service account
      in the chain. The last service account in the chain must be granted the
      `roles/iam.serviceAccountTokenCreator` role on the service account that
      is specified in the `name` field of the request.  The delegates must
      have the following format:
      `projects/-/serviceAccounts/{ACCOUNT_EMAIL_OR_UNIQUEID}`. The `-`
      wildcard character is required; replacing it with a project ID is
      invalid.
    lifetime: The desired lifetime duration of the access token in seconds.
      Must be set to a value less than or equal to 3600 (1 hour). If a value
      is not specified, the token's lifetime will be set to a default value of
      one hour.
    scope: Code to identify the scopes to be included in the OAuth 2.0 access
      token. See https://developers.google.com/identity/protocols/googlescopes
      for more information. At least one value required.
  """
    delegates = _messages.StringField(1, repeated=True)
    lifetime = _messages.StringField(2)
    scope = _messages.StringField(3, repeated=True)
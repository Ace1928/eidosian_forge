from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GenerateIdTokenRequest(_messages.Message):
    """A GenerateIdTokenRequest object.

  Fields:
    audience: The audience for the token, such as the API or account that this
      token grants access to.
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
    includeEmail: Include the service account email in the token. If set to
      `true`, the token will contain `email` and `email_verified` claims.
  """
    audience = _messages.StringField(1)
    delegates = _messages.StringField(2, repeated=True)
    includeEmail = _messages.BooleanField(3)
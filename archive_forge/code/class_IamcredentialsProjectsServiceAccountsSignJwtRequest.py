from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class IamcredentialsProjectsServiceAccountsSignJwtRequest(_messages.Message):
    """A IamcredentialsProjectsServiceAccountsSignJwtRequest object.

  Fields:
    name: The resource name of the service account for which the credentials
      are requested, in the following format:
      `projects/-/serviceAccounts/{ACCOUNT_EMAIL_OR_UNIQUEID}`. The `-`
      wildcard character is required; replacing it with a project ID is
      invalid.
    signJwtRequest: A SignJwtRequest resource to be passed as the request
      body.
  """
    name = _messages.StringField(1, required=True)
    signJwtRequest = _messages.MessageField('SignJwtRequest', 2)
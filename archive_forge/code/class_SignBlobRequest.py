from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SignBlobRequest(_messages.Message):
    """A SignBlobRequest object.

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
    payload: The bytes to sign.
  """
    delegates = _messages.StringField(1, repeated=True)
    payload = _messages.BytesField(2)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleOAuth(_messages.Message):
    """Contains information needed for generating an [OAuth
  token](https://developers.google.com/identity/protocols/OAuth2). This type
  of authorization should generally only be used when calling Google APIs
  hosted on *.googleapis.com.

  Fields:
    scope: Optional. OAuth scope to be used for generating OAuth access token.
      If not specified, "https://www.googleapis.com/auth/cloud-platform" will
      be used.
    serviceAccount: Required. [Service account
      email](https://cloud.google.com/iam/docs/service-accounts) to be used
      for generating OAuth token. The service account must be within the same
      project as the job. The caller must have iam.serviceAccounts.actAs
      permission for the service account.
  """
    scope = _messages.StringField(1)
    serviceAccount = _messages.StringField(2)
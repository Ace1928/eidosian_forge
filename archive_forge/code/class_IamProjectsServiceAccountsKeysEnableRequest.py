from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsServiceAccountsKeysEnableRequest(_messages.Message):
    """A IamProjectsServiceAccountsKeysEnableRequest object.

  Fields:
    enableServiceAccountKeyRequest: A EnableServiceAccountKeyRequest resource
      to be passed as the request body.
    name: Required. The resource name of the service account key. Use one of
      the following formats: *
      `projects/{PROJECT_ID}/serviceAccounts/{EMAIL_ADDRESS}/keys/{KEY_ID}` *
      `projects/{PROJECT_ID}/serviceAccounts/{UNIQUE_ID}/keys/{KEY_ID}` As an
      alternative, you can use the `-` wildcard character instead of the
      project ID: * `projects/-/serviceAccounts/{EMAIL_ADDRESS}/keys/{KEY_ID}`
      * `projects/-/serviceAccounts/{UNIQUE_ID}/keys/{KEY_ID}` When possible,
      avoid using the `-` wildcard character, because it can cause response
      messages to contain misleading error codes. For example, if you try to
      access the service account key
      `projects/-/serviceAccounts/fake@example.com/keys/fake-key`, which does
      not exist, the response contains an HTTP `403 Forbidden` error instead
      of a `404 Not Found` error.
  """
    enableServiceAccountKeyRequest = _messages.MessageField('EnableServiceAccountKeyRequest', 1)
    name = _messages.StringField(2, required=True)
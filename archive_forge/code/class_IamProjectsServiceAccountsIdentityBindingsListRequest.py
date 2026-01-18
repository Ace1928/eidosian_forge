from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsServiceAccountsIdentityBindingsListRequest(_messages.Message):
    """A IamProjectsServiceAccountsIdentityBindingsListRequest object.

  Fields:
    name: The resource name of the service account. Use one of the following
      formats: * `projects/{PROJECT_ID}/serviceAccounts/{EMAIL_ADDRESS}` *
      `projects/{PROJECT_ID}/serviceAccounts/{UNIQUE_ID}` As an alternative,
      you can use the `-` wildcard character instead of the project ID: *
      `projects/-/serviceAccounts/{EMAIL_ADDRESS}` *
      `projects/-/serviceAccounts/{UNIQUE_ID}` When possible, avoid using the
      `-` wildcard character, because it can cause response messages to
      contain misleading error codes. For example, if you try to access the
      service account `projects/-/serviceAccounts/fake@example.com`, which
      does not exist, the response contains an HTTP `403 Forbidden` error
      instead of a `404 Not Found` error.
  """
    name = _messages.StringField(1, required=True)
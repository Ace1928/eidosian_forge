from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CreateServiceAccountRequest(_messages.Message):
    """The service account create request.

  Fields:
    accountId: Required. The account id that is used to generate the service
      account email address and a stable unique id. It is unique within a
      project, must be 1-63 characters long, and match the regular expression
      `[a-z]([-a-z0-9]*[a-z0-9])` to comply with RFC1035.
    serviceAccount: The ServiceAccount resource to create. Currently, only the
      following values are user assignable: `display_name` .
  """
    accountId = _messages.StringField(1)
    serviceAccount = _messages.MessageField('ServiceAccount', 2)
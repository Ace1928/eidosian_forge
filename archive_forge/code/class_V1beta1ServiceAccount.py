from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class V1beta1ServiceAccount(_messages.Message):
    """A service account in the Identity and Access Management API.

  Fields:
    email: The email address of the service account.
    iamAccountName: Deprecated. See b/136209818.
    name: P4 SA resource name.  An example name would be: `services/servicecon
      sumermanagement.googleapis.com/projects/123/serviceAccounts/default`
    tag: The P4 SA configuration tag. This must be defined in
      activation_grants. If not specified when creating the account, the tag
      is set to "default".
    uniqueId: The unique and stable id of the service account.
  """
    email = _messages.StringField(1)
    iamAccountName = _messages.StringField(2)
    name = _messages.StringField(3)
    tag = _messages.StringField(4)
    uniqueId = _messages.StringField(5)
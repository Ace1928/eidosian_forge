from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudshellUsersEnvironmentsAddPublicKeyRequest(_messages.Message):
    """A CloudshellUsersEnvironmentsAddPublicKeyRequest object.

  Fields:
    addPublicKeyRequest: A AddPublicKeyRequest resource to be passed as the
      request body.
    environment: Environment this key should be added to, e.g.
      `users/me/environments/default`.
  """
    addPublicKeyRequest = _messages.MessageField('AddPublicKeyRequest', 1)
    environment = _messages.StringField(2, required=True)
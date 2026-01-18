from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsloginUsersSshPublicKeysCreateRequest(_messages.Message):
    """A OsloginUsersSshPublicKeysCreateRequest object.

  Fields:
    parent: Required. The unique ID for the user in format `users/{user}`.
    sshPublicKey: A SshPublicKey resource to be passed as the request body.
  """
    parent = _messages.StringField(1, required=True)
    sshPublicKey = _messages.MessageField('SshPublicKey', 2)
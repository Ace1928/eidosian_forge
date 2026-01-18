from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ImportSshPublicKeyResponse(_messages.Message):
    """A response message for importing an SSH public key.

  Fields:
    details: Detailed information about import results.
    loginProfile: The login profile information for the user.
  """
    details = _messages.StringField(1)
    loginProfile = _messages.MessageField('LoginProfile', 2)
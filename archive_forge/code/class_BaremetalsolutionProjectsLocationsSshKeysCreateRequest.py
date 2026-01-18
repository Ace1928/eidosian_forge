from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BaremetalsolutionProjectsLocationsSshKeysCreateRequest(_messages.Message):
    """A BaremetalsolutionProjectsLocationsSshKeysCreateRequest object.

  Fields:
    parent: Required. The parent containing the SSH keys.
    sSHKey: A SSHKey resource to be passed as the request body.
    sshKeyId: Required. The ID to use for the key, which will become the final
      component of the key's resource name. This value must match the regex:
      [a-zA-Z0-9@.\\-_]{1,64}
  """
    parent = _messages.StringField(1, required=True)
    sSHKey = _messages.MessageField('SSHKey', 2)
    sshKeyId = _messages.StringField(3)
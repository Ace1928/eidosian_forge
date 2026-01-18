from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AddSecretVersionRequest(_messages.Message):
    """Request message for SecretManagerService.AddSecretVersion.

  Fields:
    payload: Required. The secret payload of the SecretVersion.
  """
    payload = _messages.MessageField('SecretPayload', 1)
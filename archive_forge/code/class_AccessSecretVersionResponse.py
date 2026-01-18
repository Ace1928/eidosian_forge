from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccessSecretVersionResponse(_messages.Message):
    """Response message for SecretManagerService.AccessSecretVersion.

  Fields:
    name: The resource name of the SecretVersion in the format
      `projects/*/secrets/*/versions/*` or
      `projects/*/locations/*/secrets/*/versions/*`.
    payload: Secret payload
  """
    name = _messages.StringField(1)
    payload = _messages.MessageField('SecretPayload', 2)
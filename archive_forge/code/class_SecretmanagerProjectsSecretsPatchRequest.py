from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecretmanagerProjectsSecretsPatchRequest(_messages.Message):
    """A SecretmanagerProjectsSecretsPatchRequest object.

  Fields:
    name: Output only. The resource name of the Secret in the format
      `projects/*/secrets/*`.
    secret: A Secret resource to be passed as the request body.
    updateMask: Required. Specifies the fields to be updated.
  """
    name = _messages.StringField(1, required=True)
    secret = _messages.MessageField('Secret', 2)
    updateMask = _messages.StringField(3)
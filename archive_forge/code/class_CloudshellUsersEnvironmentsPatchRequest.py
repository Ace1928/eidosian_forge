from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudshellUsersEnvironmentsPatchRequest(_messages.Message):
    """A CloudshellUsersEnvironmentsPatchRequest object.

  Fields:
    environment: A Environment resource to be passed as the request body.
    name: Name of the resource to be updated, for example
      `users/me/environments/default` or
      `users/someone@example.com/environments/default`.
    updateMask: Mask specifying which fields in the environment should be
      updated.
  """
    environment = _messages.MessageField('Environment', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)
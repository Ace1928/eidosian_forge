from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudshellUsersEnvironmentsStartRequest(_messages.Message):
    """A CloudshellUsersEnvironmentsStartRequest object.

  Fields:
    name: Name of the resource that should be started, for example
      `users/me/environments/default` or
      `users/someone@example.com/environments/default`.
    startEnvironmentRequest: A StartEnvironmentRequest resource to be passed
      as the request body.
  """
    name = _messages.StringField(1, required=True)
    startEnvironmentRequest = _messages.MessageField('StartEnvironmentRequest', 2)
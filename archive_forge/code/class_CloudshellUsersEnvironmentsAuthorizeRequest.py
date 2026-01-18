from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudshellUsersEnvironmentsAuthorizeRequest(_messages.Message):
    """A CloudshellUsersEnvironmentsAuthorizeRequest object.

  Fields:
    authorizeEnvironmentRequest: A AuthorizeEnvironmentRequest resource to be
      passed as the request body.
    name: Name of the resource that should receive the credentials, for
      example `users/me/environments/default` or
      `users/someone@example.com/environments/default`.
  """
    authorizeEnvironmentRequest = _messages.MessageField('AuthorizeEnvironmentRequest', 1)
    name = _messages.StringField(2, required=True)
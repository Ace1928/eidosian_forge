from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsOauthClientsUndeleteRequest(_messages.Message):
    """A IamProjectsLocationsOauthClientsUndeleteRequest object.

  Fields:
    name: Required. The name of the oauth client to undelete. Format:
      `projects/{project}/locations/{location}/oauthClients/{oauth_client}`.
    undeleteOauthClientRequest: A UndeleteOauthClientRequest resource to be
      passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    undeleteOauthClientRequest = _messages.MessageField('UndeleteOauthClientRequest', 2)
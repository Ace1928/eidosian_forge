from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsOauthClientsCredentialsGetRequest(_messages.Message):
    """A IamProjectsLocationsOauthClientsCredentialsGetRequest object.

  Fields:
    name: Required. The name of the oauth client credential to retrieve.
      Format: `projects/{project}/locations/{location}/oauthClients/{oauth_cli
      ent}/credentials/{credential}`.
  """
    name = _messages.StringField(1, required=True)
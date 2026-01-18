from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsOauthClientsGetRequest(_messages.Message):
    """A IamProjectsLocationsOauthClientsGetRequest object.

  Fields:
    name: Required. The name of the oauth client to retrieve. Format:
      `projects/{project}/locations/{location}/oauthClients/{oauth_client}`.
  """
    name = _messages.StringField(1, required=True)
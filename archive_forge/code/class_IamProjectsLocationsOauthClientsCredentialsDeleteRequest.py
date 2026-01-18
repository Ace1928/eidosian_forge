from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsOauthClientsCredentialsDeleteRequest(_messages.Message):
    """A IamProjectsLocationsOauthClientsCredentialsDeleteRequest object.

  Fields:
    name: Required. The name of the oauth client credential to delete. Format:
      `projects/{project}/locations/{location}/oauthClients/{oauth_client}/cre
      dentials/{credential}`.
    validateOnly: Optional. If set, validate the request and preview the
      response, but do not actually post it.
  """
    name = _messages.StringField(1, required=True)
    validateOnly = _messages.BooleanField(2)
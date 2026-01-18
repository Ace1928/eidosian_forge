from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListUserWorkloadsSecretsResponse(_messages.Message):
    """The user workloads Secrets for a given environment.

  Fields:
    nextPageToken: The page token used to query for the next page if one
      exists.
    userWorkloadsSecrets: The list of Secrets returned by a
      ListUserWorkloadsSecretsRequest.
  """
    nextPageToken = _messages.StringField(1)
    userWorkloadsSecrets = _messages.MessageField('UserWorkloadsSecret', 2, repeated=True)
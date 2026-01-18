from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TypeProvidersListResponse(_messages.Message):
    """A response that returns all Type Providers supported by Deployment
  Manager

  Fields:
    nextPageToken: A token used to continue a truncated list request.
    typeProviders: Output only. A list of resource type providers supported by
      Deployment Manager.
  """
    nextPageToken = _messages.StringField(1)
    typeProviders = _messages.MessageField('TypeProvider', 2, repeated=True)
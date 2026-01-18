from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListProvidersResponse(_messages.Message):
    """Response message for Connectors.ListProviders.

  Fields:
    nextPageToken: Next page token.
    providers: A list of providers.
    unreachable: Locations that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    providers = _messages.MessageField('Provider', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListSQLIntegrationsResponse(_messages.Message):
    """ListSQLIntegrationsResponse is the response message for
  ListSQLIntegrations method.

  Fields:
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
    sqlIntegrations: A list of SQLIntegrations of a domain.
    unreachable: A list of locations that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    sqlIntegrations = _messages.MessageField('SQLIntegration', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)
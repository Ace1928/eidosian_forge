from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListTrustConfigsResponse(_messages.Message):
    """Response for the `ListTrustConfigs` method.

  Fields:
    nextPageToken: If there might be more results than those appearing in this
      response, then `next_page_token` is included. To get the next set of
      results, call this method again using the value of `next_page_token` as
      `page_token`.
    trustConfigs: A list of TrustConfigs for the parent resource.
    unreachable: Locations that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    trustConfigs = _messages.MessageField('TrustConfig', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListEndpointPoliciesResponse(_messages.Message):
    """Response returned by the ListEndpointPolicies method.

  Fields:
    endpointPolicies: List of EndpointPolicy resources.
    nextPageToken: If there might be more results than those appearing in this
      response, then `next_page_token` is included. To get the next set of
      results, call this method again using the value of `next_page_token` as
      `page_token`.
  """
    endpointPolicies = _messages.MessageField('EndpointPolicy', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
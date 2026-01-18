from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListServiceLbPoliciesResponse(_messages.Message):
    """Response returned by the ListServiceLbPolicies method.

  Fields:
    nextPageToken: If there might be more results than those appearing in this
      response, then `next_page_token` is included. To get the next set of
      results, call this method again using the value of `next_page_token` as
      `page_token`.
    serviceLbPolicies: List of ServiceLbPolicy resources.
  """
    nextPageToken = _messages.StringField(1)
    serviceLbPolicies = _messages.MessageField('ServiceLbPolicy', 2, repeated=True)
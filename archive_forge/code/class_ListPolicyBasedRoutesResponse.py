from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListPolicyBasedRoutesResponse(_messages.Message):
    """Response for PolicyBasedRouting.ListPolicyBasedRoutes method.

  Fields:
    nextPageToken: The next pagination token in the List response. It should
      be used as page_token for the following request. An empty value means no
      more result.
    policyBasedRoutes: Policy-based routes to be returned.
    unreachable: Locations that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    policyBasedRoutes = _messages.MessageField('PolicyBasedRoute', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)
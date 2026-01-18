from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListPoliciesResponse(_messages.Message):
    """Response returned by policies.list method.

  Fields:
    nextPageToken: Next page token. Provide this to retrieve the subsequent
      page. When paginating, all other parameters (except page_size) provided
      to policies.list must match the call that provided the page token.
    policies: The list of Policies.
    unreachable: Represents missing potential additional resources.
  """
    nextPageToken = _messages.StringField(1)
    policies = _messages.MessageField('Policy', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListAccessPoliciesResponse(_messages.Message):
    """A response to `ListAccessPoliciesRequest`.

  Fields:
    accessPolicies: List of the AccessPolicy instances.
    nextPageToken: The pagination token to retrieve the next page of results.
      If the value is empty, no further results remain.
  """
    accessPolicies = _messages.MessageField('AccessPolicy', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
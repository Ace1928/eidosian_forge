from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListPlatformPoliciesResponse(_messages.Message):
    """Response message for
  PlatformPolicyManagementService.ListPlatformPolicies.

  Fields:
    nextPageToken: A token to retrieve the next page of results. Pass this
      value in the ListPlatformPoliciesRequest.page_token field in the
      subsequent call to the `ListPlatformPolicies` method to retrieve the
      next page of results.
    platformPolicies: The list of platform policies.
  """
    nextPageToken = _messages.StringField(1)
    platformPolicies = _messages.MessageField('PlatformPolicy', 2, repeated=True)
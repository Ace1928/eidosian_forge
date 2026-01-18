from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GetPolicyDetailsRequest(_messages.Message):
    """The request to get the current policy and the policies on the inherited
  resources the user has access to.

  Fields:
    fullResourcePath: REQUIRED: The full resource path of the current policy
      being requested, e.g., `//dataflow.googleapis.com/projects/../jobs/..`.
    pageSize: Limit on the number of policies to include in the response.
      Further accounts can subsequently be obtained by including the
      GetPolicyDetailsResponse.next_page_token in a subsequent request. If
      zero, the default page size 20 will be used. Must be given a value in
      range [0, 100], otherwise an invalid argument error will be returned.
    pageToken: Optional pagination token returned in an earlier
      GetPolicyDetailsResponse.next_page_token response.
  """
    fullResourcePath = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
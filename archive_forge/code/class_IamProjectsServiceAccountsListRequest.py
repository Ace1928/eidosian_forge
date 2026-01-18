from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class IamProjectsServiceAccountsListRequest(_messages.Message):
    """A IamProjectsServiceAccountsListRequest object.

  Fields:
    name: Required. The resource name of the project associated with the
      service accounts, such as `projects/my-project-123`.
    pageSize: Optional limit on the number of service accounts to include in
      the response. Further accounts can subsequently be obtained by including
      the ListServiceAccountsResponse.next_page_token in a subsequent request.
    pageToken: Optional pagination token returned in an earlier
      ListServiceAccountsResponse.next_page_token.
    removeDeletedServiceAccounts: Do not list service accounts deleted from
      Gaia. <b><font color="red">DO NOT INCLUDE IN EXTERNAL
      DOCUMENTATION</font></b>.
  """
    name = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    removeDeletedServiceAccounts = _messages.BooleanField(4)
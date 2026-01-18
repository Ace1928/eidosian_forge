from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityCustomersUserinvitationsListRequest(_messages.Message):
    """A CloudidentityCustomersUserinvitationsListRequest object.

  Fields:
    filter: Optional. A query string for filtering `UserInvitation` results by
      their current state, in the format: `"state=='invited'"`.
    orderBy: Optional. The sort order of the list results. You can sort the
      results in descending order based on either email or last update
      timestamp but not both, using `order_by="email desc"`. Currently,
      sorting is supported for `update_time asc`, `update_time desc`, `email
      asc`, and `email desc`. If not specified, results will be returned based
      on `email asc` order.
    pageSize: Optional. The maximum number of UserInvitation resources to
      return. If unspecified, at most 100 resources will be returned. The
      maximum value is 200; values above 200 will be set to 200.
    pageToken: Optional. A page token, received from a previous
      `ListUserInvitations` call. Provide this to retrieve the subsequent
      page. When paginating, all other parameters provided to `ListBooks` must
      match the call that provided the page token.
    parent: Required. The customer ID of the Google Workspace or Cloud
      Identity account the UserInvitation resources are associated with.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)
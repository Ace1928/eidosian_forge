from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AccessapprovalFoldersApprovalRequestsListRequest(_messages.Message):
    """A AccessapprovalFoldersApprovalRequestsListRequest object.

  Fields:
    filter: A filter on the type of approval requests to retrieve. Must be one
      of the following values: * [not set]: Requests that are pending or have
      active approvals. * ALL: All requests. * PENDING: Only pending requests.
      * ACTIVE: Only active (i.e. currently approved) requests. * DISMISSED:
      Only requests that have been dismissed, or requests that are not
      approved and past expiration. * EXPIRED: Only requests that have been
      approved, and the approval has expired. * HISTORY: Active, dismissed and
      expired requests.
    pageSize: Requested page size.
    pageToken: A token identifying the page of results to return.
    parent: The parent resource. This may be "projects/{project}",
      "folders/{folder}", or "organizations/{organization}".
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)
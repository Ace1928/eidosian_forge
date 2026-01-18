from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudbillingBillingAccountsProjectsListRequest(_messages.Message):
    """A CloudbillingBillingAccountsProjectsListRequest object.

  Fields:
    name: Required. The resource name of the billing account associated with
      the projects that you want to list. For example,
      `billingAccounts/012345-567890-ABCDEF`.
    pageSize: Requested page size. The maximum page size is 100; this is also
      the default.
    pageToken: A token identifying a page of results to be returned. This
      should be a `next_page_token` value returned from a previous
      `ListProjectBillingInfo` call. If unspecified, the first page of results
      is returned.
  """
    name = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudcommerceconsumerprocurementBillingAccountsOrdersListRequest(_messages.Message):
    """A CloudcommerceconsumerprocurementBillingAccountsOrdersListRequest
  object.

  Fields:
    filter: Filter that you can use to limit the list request. A query string
      that can match a selected set of attributes with string values. For
      example, `display_name=abc`. Supported query attributes are *
      `display_name` If the query contains special characters other than
      letters, underscore, or digits, the phrase must be quoted with double
      quotes. For example, `display_name="foo:bar"`, where the display name
      needs to be quoted because it contains special character colon. Queries
      can be combined with `OR`, and `NOT` to form more complex queries. You
      can also group them to force a desired evaluation order. For example,
      `display_name=abc OR display_name=def`.
    pageSize: The maximum number of entries requested. The default page size
      is 25 and the maximum page size is 200.
    pageToken: The token for fetching the next page.
    parent: Required. The parent resource to query for orders. This field has
      the form `billingAccounts/{billing-account-id}`.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)
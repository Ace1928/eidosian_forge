from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudcommerceconsumerprocurementProjectsFreeTrialsListRequest(_messages.Message):
    """A CloudcommerceconsumerprocurementProjectsFreeTrialsListRequest object.

  Fields:
    filter: The filter that can be used to limit the list request. The filter
      is a query string that can match a selected set of attributes with
      string values. For example `product_external_name=1234-5678-ABCD-EFG`.
      Supported query attributes are * `product_external_name` * `provider` *
      `service` Service queries have the format:
      `service="services/{serviceID}"` where serviceID is the
      OnePlatformServiceId. If the query contains special characters other
      than letters, underscore, or digits, the phrase must be quoted with
      double quotes. For example, `product_external_name="foo:bar"`, where the
      product name needs to be quoted because it contains special character
      colon. Queries can be combined with `AND`, `OR`, and `NOT` to form more
      complex queries. They can also be grouped to force a desired evaluation
      order. For example, `provider=providers/E-1234 OR
      provider=providers/5678 AND NOT (product_external_name=foo-product)`.
      Connective `AND` can be omitted between two predicates. For example
      `provider=providers/E-1234 product_external_name=foo` is equivalent to
      `provider=providers/E-1234 AND product_external_name=foo`.
    pageSize: The maximum number of entries that are requested. The default
      page size is 25 and the maximum page size is 200.
    pageToken: The token for fetching the next page.
    parent: Required. The parent resource to query for FreeTrials. Currently
      the only parent supported is "projects/{project-id}".
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)
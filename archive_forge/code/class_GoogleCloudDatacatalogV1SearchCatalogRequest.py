from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1SearchCatalogRequest(_messages.Message):
    """Request message for SearchCatalog.

  Fields:
    adminSearch: Optional. If set, use searchAll permission granted on
      organizations from `include_org_ids` and projects from
      `include_project_ids` instead of the fine grained per resource
      permissions when filtering the search results. The only allowed
      `order_by` criteria for admin_search mode is `default`. Using this flags
      guarantees a full recall of the search results.
    orderBy: Specifies the order of results. Currently supported case-
      sensitive values are: * `relevance` that can only be descending *
      `last_modified_timestamp [asc|desc]` with descending (`desc`) as default
      * `default` that can only be descending Search queries don't guarantee
      full recall. Results that match your query might not be returned, even
      in subsequent result pages. Additionally, returned (and not returned)
      results can vary if you repeat search queries. If you are experiencing
      recall issues and you don't have to fetch the results in any specific
      order, consider setting this parameter to `default`. If this parameter
      is omitted, it defaults to the descending `relevance`.
    pageSize: Upper bound on the number of results you can get in a single
      response. Can't be negative or 0, defaults to 10 in this case. The
      maximum number is 1000. If exceeded, throws an "invalid argument"
      exception.
    pageToken: Optional. Pagination token that, if specified, returns the next
      page of search results. If empty, returns the first page. This token is
      returned in the SearchCatalogResponse.next_page_token field of the
      response to a previous SearchCatalogRequest call.
    query: Optional. The query string with a minimum of 3 characters and
      specific syntax. For more information, see [Data Catalog search
      syntax](https://cloud.google.com/data-catalog/docs/how-to/search-
      reference). An empty query string returns all data assets (in the
      specified scope) that you have access to. A query string can be a simple
      `xyz` or qualified by predicates: * `name:x` * `column:y` *
      `description:z`
    scope: Required. The scope of this search request. The `scope` is invalid
      if `include_org_ids`, `include_project_ids` are empty AND
      `include_gcp_public_datasets` is set to `false`. In this case, the
      request returns an error.
  """
    adminSearch = _messages.BooleanField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    query = _messages.StringField(5)
    scope = _messages.MessageField('GoogleCloudDatacatalogV1SearchCatalogRequestScope', 6)
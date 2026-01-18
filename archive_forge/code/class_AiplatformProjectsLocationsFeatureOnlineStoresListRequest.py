from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeatureOnlineStoresListRequest(_messages.Message):
    """A AiplatformProjectsLocationsFeatureOnlineStoresListRequest object.

  Fields:
    filter: Lists the FeatureOnlineStores that match the filter expression.
      The following fields are supported: * `create_time`: Supports `=`, `!=`,
      `<`, `>`, `<=`, and `>=` comparisons. Values must be in RFC 3339 format.
      * `update_time`: Supports `=`, `!=`, `<`, `>`, `<=`, and `>=`
      comparisons. Values must be in RFC 3339 format. * `labels`: Supports
      key-value equality and key presence. Examples: * `create_time >
      "2020-01-01" OR update_time > "2020-01-01"` FeatureOnlineStores created
      or updated after 2020-01-01. * `labels.env = "prod"` FeatureOnlineStores
      with label "env" set to "prod".
    orderBy: A comma-separated list of fields to order by, sorted in ascending
      order. Use "desc" after a field name for descending. Supported Fields: *
      `create_time` * `update_time`
    pageSize: The maximum number of FeatureOnlineStores to return. The service
      may return fewer than this value. If unspecified, at most 100
      FeatureOnlineStores will be returned. The maximum value is 100; any
      value greater than 100 will be coerced to 100.
    pageToken: A page token, received from a previous
      FeatureOnlineStoreAdminService.ListFeatureOnlineStores call. Provide
      this to retrieve the subsequent page. When paginating, all other
      parameters provided to
      FeatureOnlineStoreAdminService.ListFeatureOnlineStores must match the
      call that provided the page token.
    parent: Required. The resource name of the Location to list
      FeatureOnlineStores. Format: `projects/{project}/locations/{location}`
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)
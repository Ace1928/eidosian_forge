from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudquotasOrganizationsLocationsQuotaPreferencesListRequest(_messages.Message):
    """A CloudquotasOrganizationsLocationsQuotaPreferencesListRequest object.

  Fields:
    filter: Optional. Filter result QuotaPreferences by their state, type,
      create/update time range. Example filters: `reconciling=true AND
      request_type=CLOUD_CONSOLE`, `reconciling=true OR
      creation_time>2022-12-03T10:30:00`
    orderBy: Optional. How to order of the results. By default, the results
      are ordered by create time. Example orders: `quota_id`, `service,
      create_time`
    pageSize: Optional. Requested page size. Server may return fewer items
      than requested. If unspecified, server will pick an appropriate default.
    pageToken: Optional. A token identifying a page of results the server
      should return.
    parent: Required. Parent value of QuotaPreference resources. Listing
      across different resource containers (such as 'projects/-') is not
      allowed. When the value starts with 'folders' or 'organizations', it
      lists the QuotaPreferences for org quotas in the container. It does not
      list the QuotaPreferences in the descendant projects of the container.
      Example parents: `projects/123/locations/global`
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)
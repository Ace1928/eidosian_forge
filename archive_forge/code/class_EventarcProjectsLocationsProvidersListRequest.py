from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventarcProjectsLocationsProvidersListRequest(_messages.Message):
    """A EventarcProjectsLocationsProvidersListRequest object.

  Fields:
    filter: The filter field that the list request will filter on.
    orderBy: The sorting order of the resources returned. Value should be a
      comma-separated list of fields. The default sorting oder is ascending.
      To specify descending order for a field, append a `desc` suffix; for
      example: `name desc, _id`.
    pageSize: The maximum number of providers to return on each page.
    pageToken: The page token; provide the value from the `next_page_token`
      field in a previous `ListProviders` call to retrieve the subsequent
      page. When paginating, all other parameters provided to `ListProviders`
      must match the call that provided the page token.
    parent: Required. The parent of the provider to get.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)
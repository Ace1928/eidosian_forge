from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsEntryGroupsTagsListRequest(_messages.Message):
    """A DatacatalogProjectsLocationsEntryGroupsTagsListRequest object.

  Fields:
    pageSize: The maximum number of tags to return. Default is 10. Maximum
      limit is 1000.
    pageToken: Pagination token that specifies the next page to return. If
      empty, the first page is returned.
    parent: Required. The name of the Data Catalog resource to list the tags
      of. The resource can be an Entry or an EntryGroup (without
      `/entries/{entries}` at the end).
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
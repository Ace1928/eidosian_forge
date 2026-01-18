from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsEntryGroupsEntriesListRequest(_messages.Message):
    """A DatacatalogProjectsLocationsEntryGroupsEntriesListRequest object.

  Fields:
    pageSize: The maximum number of items to return. Default is 10. Maximum
      limit is 1000. Throws an invalid argument if `page_size` is more than
      1000.
    pageToken: Pagination token that specifies the next page to return. If
      empty, the first page is returned.
    parent: Required. The name of the entry group that contains the entries to
      list. Can be provided in URL format.
    readMask: The fields to return for each entry. If empty or omitted, all
      fields are returned. For example, to return a list of entries with only
      the `name` field, set `read_mask` to only one path with the `name`
      value.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    readMask = _messages.StringField(4)
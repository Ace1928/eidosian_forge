from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsEntryTypesListRequest(_messages.Message):
    """A DataplexProjectsLocationsEntryTypesListRequest object.

  Fields:
    filter: Optional. Filter request. Filters are case-sensitive. The
      following formats are supported:labels.key1 = "value1" labels:key1 name
      = "value" These restrictions can be coinjoined with AND, OR and NOT
      conjunctions.
    orderBy: Optional. Order by fields (name or create_time) for the result.
      If not specified, the ordering is undefined.
    pageSize: Optional. Maximum number of EntryTypes to return. The service
      may return fewer than this value. If unspecified, at most 10 EntryTypes
      will be returned. The maximum value is 1000; values above 1000 will be
      coerced to 1000.
    pageToken: Optional. Page token received from a previous ListEntryTypes
      call. Provide this to retrieve the subsequent page. When paginating, all
      other parameters provided to ListEntryTypes must match the call that
      provided the page token.
    parent: Required. The resource name of the EntryType location, of the
      form: projects/{project_number}/locations/{location_id} where
      location_id refers to a GCP region.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)
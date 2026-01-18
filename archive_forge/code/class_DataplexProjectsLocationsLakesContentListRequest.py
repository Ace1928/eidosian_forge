from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsLakesContentListRequest(_messages.Message):
    """A DataplexProjectsLocationsLakesContentListRequest object.

  Fields:
    filter: Optional. Filter request. Filters are case-sensitive. The
      following formats are supported:labels.key1 = "value1" labels:key1 type
      = "NOTEBOOK" type = "SQL_SCRIPT"These restrictions can be coinjoined
      with AND, OR and NOT conjunctions.
    pageSize: Optional. Maximum number of content to return. The service may
      return fewer than this value. If unspecified, at most 10 content will be
      returned. The maximum value is 1000; values above 1000 will be coerced
      to 1000.
    pageToken: Optional. Page token received from a previous ListContent call.
      Provide this to retrieve the subsequent page. When paginating, all other
      parameters provided to ListContent must match the call that provided the
      page token.
    parent: Required. The resource name of the parent lake:
      projects/{project_id}/locations/{location_id}/lakes/{lake_id}
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)
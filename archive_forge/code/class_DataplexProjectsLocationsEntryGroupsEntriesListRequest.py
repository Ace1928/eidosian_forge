from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsEntryGroupsEntriesListRequest(_messages.Message):
    """A DataplexProjectsLocationsEntryGroupsEntriesListRequest object.

  Fields:
    filter: Optional. A filter on the entries to return. Filters are case-
      sensitive. The request can be filtered by the following fields:
      entry_type, entry_source.display_name. The comparison operators are =,
      !=, <, >, <=, >= (strings are compared according to lexical order) The
      logical operators AND, OR, NOT can be used in the filter. Wildcard "*"
      can be used, but for entry_type the full project id or number needs to
      be provided. Example filter expressions:
      "entry_source.display_name=AnExampleDisplayName"
      "entry_type=projects/example-
      project/locations/global/entryTypes/example-entry_type"
      "entry_type=projects/example-project/locations/us/entryTypes/a* OR
      entry_type=projects/another-project/locations/*" "NOT
      entry_source.display_name=AnotherExampleDisplayName"
    pageSize: A integer attribute.
    pageToken: Optional. The pagination token returned by a previous request.
    parent: Required. The resource name of the parent Entry Group:
      projects/{project}/locations/{location}/entryGroups/{entry_group}.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)
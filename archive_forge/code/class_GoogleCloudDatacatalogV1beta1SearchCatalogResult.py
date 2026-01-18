from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1beta1SearchCatalogResult(_messages.Message):
    """A result that appears in the response of a search request. Each result
  captures details of one entry that matches the search.

  Enums:
    SearchResultTypeValueValuesEnum: Type of the search result. This field can
      be used to determine which Get method to call to fetch the full
      resource.

  Fields:
    linkedResource: The full name of the cloud resource the entry belongs to.
      See:
      https://cloud.google.com/apis/design/resource_names#full_resource_name.
      Example: * `//bigquery.googleapis.com/projects/projectId/datasets/datase
      tId/tables/tableId`
    modifyTime: Last-modified timestamp of the entry from the managing system.
    relativeResourceName: The relative resource name of the resource in URL
      format. Examples: * `projects/{project_id}/locations/{location_id}/entry
      Groups/{entry_group_id}/entries/{entry_id}` *
      `projects/{project_id}/tagTemplates/{tag_template_id}`
    searchResultSubtype: Sub-type of the search result. This is a dot-
      delimited description of the resource's full type, and is the same as
      the value callers would provide in the "type" search facet. Examples:
      `entry.table`, `entry.dataStream`, `tagTemplate`.
    searchResultType: Type of the search result. This field can be used to
      determine which Get method to call to fetch the full resource.
  """

    class SearchResultTypeValueValuesEnum(_messages.Enum):
        """Type of the search result. This field can be used to determine which
    Get method to call to fetch the full resource.

    Values:
      SEARCH_RESULT_TYPE_UNSPECIFIED: Default unknown type.
      ENTRY: An Entry.
      TAG_TEMPLATE: A TagTemplate.
      ENTRY_GROUP: An EntryGroup.
    """
        SEARCH_RESULT_TYPE_UNSPECIFIED = 0
        ENTRY = 1
        TAG_TEMPLATE = 2
        ENTRY_GROUP = 3
    linkedResource = _messages.StringField(1)
    modifyTime = _messages.StringField(2)
    relativeResourceName = _messages.StringField(3)
    searchResultSubtype = _messages.StringField(4)
    searchResultType = _messages.EnumField('SearchResultTypeValueValuesEnum', 5)
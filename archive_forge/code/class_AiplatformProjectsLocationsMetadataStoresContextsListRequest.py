from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsMetadataStoresContextsListRequest(_messages.Message):
    """A AiplatformProjectsLocationsMetadataStoresContextsListRequest object.

  Fields:
    filter: Filter specifying the boolean condition for the Contexts to
      satisfy in order to be part of the result set. The syntax to define
      filter query is based on https://google.aip.dev/160. Following are the
      supported set of filters: * **Attribute filtering**: For example:
      `display_name = "test"`. Supported fields include: `name`,
      `display_name`, `schema_title`, `create_time`, and `update_time`. Time
      fields, such as `create_time` and `update_time`, require values
      specified in RFC-3339 format. For example: `create_time =
      "2020-11-19T11:30:00-04:00"`. * **Metadata field**: To filter on
      metadata fields use traversal operation as follows: `metadata..`. For
      example: `metadata.field_1.number_value = 10.0`. In case the field name
      contains special characters (such as colon), one can embed it inside
      double quote. For example: `metadata."field:1".number_value = 10.0` *
      **Parent Child filtering**: To filter Contexts based on parent-child
      relationship use the HAS operator as follows: ``` parent_contexts:
      "projects//locations//metadataStores//contexts/" child_contexts:
      "projects//locations//metadataStores//contexts/" ``` Each of the above
      supported filters can be combined together using logical operators
      (`AND` & `OR`). Maximum nested expression depth allowed is 5. For
      example: `display_name = "test" AND metadata.field1.bool_value = true`.
    orderBy: How the list of messages is ordered. Specify the values to order
      by and an ordering operation. The default sorting order is ascending. To
      specify descending order for a field, users append a " desc" suffix; for
      example: "foo desc, bar". Subfields are specified with a `.` character,
      such as foo.bar. see https://google.aip.dev/132#ordering for more
      details.
    pageSize: The maximum number of Contexts to return. The service may return
      fewer. Must be in range 1-1000, inclusive. Defaults to 100.
    pageToken: A page token, received from a previous
      MetadataService.ListContexts call. Provide this to retrieve the
      subsequent page. When paginating, all other provided parameters must
      match the call that provided the page token. (Otherwise the request will
      fail with INVALID_ARGUMENT error.)
    parent: Required. The MetadataStore whose Contexts should be listed.
      Format:
      `projects/{project}/locations/{location}/metadataStores/{metadatastore}`
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)
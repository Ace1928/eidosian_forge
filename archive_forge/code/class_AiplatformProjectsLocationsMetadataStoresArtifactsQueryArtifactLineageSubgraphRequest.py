from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsMetadataStoresArtifactsQueryArtifactLineageSubgraphRequest(_messages.Message):
    """A AiplatformProjectsLocationsMetadataStoresArtifactsQueryArtifactLineage
  SubgraphRequest object.

  Fields:
    artifact: Required. The resource name of the Artifact whose Lineage needs
      to be retrieved as a LineageSubgraph. Format: `projects/{project}/locati
      ons/{location}/metadataStores/{metadatastore}/artifacts/{artifact}` The
      request may error with FAILED_PRECONDITION if the number of Artifacts,
      the number of Executions, or the number of Events that would be returned
      for the Context exceeds 1000.
    filter: Filter specifying the boolean condition for the Artifacts to
      satisfy in order to be part of the Lineage Subgraph. The syntax to
      define filter query is based on https://google.aip.dev/160. The
      supported set of filters include the following: * **Attribute
      filtering**: For example: `display_name = "test"` Supported fields
      include: `name`, `display_name`, `uri`, `state`, `schema_title`,
      `create_time`, and `update_time`. Time fields, such as `create_time` and
      `update_time`, require values specified in RFC-3339 format. For example:
      `create_time = "2020-11-19T11:30:00-04:00"` * **Metadata field**: To
      filter on metadata fields use traversal operation as follows:
      `metadata..`. For example: `metadata.field_1.number_value = 10.0` In
      case the field name contains special characters (such as colon), one can
      embed it inside double quote. For example:
      `metadata."field:1".number_value = 10.0` Each of the above supported
      filter types can be combined together using logical operators (`AND` &
      `OR`). Maximum nested expression depth allowed is 5. For example:
      `display_name = "test" AND metadata.field1.bool_value = true`.
    maxHops: Specifies the size of the lineage graph in terms of number of
      hops from the specified artifact. Negative Value: INVALID_ARGUMENT error
      is returned 0: Only input artifact is returned. No value: Transitive
      closure is performed to return the complete graph.
  """
    artifact = _messages.StringField(1, required=True)
    filter = _messages.StringField(2)
    maxHops = _messages.IntegerField(3, variant=_messages.Variant.INT32)
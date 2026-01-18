from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SchemaConfig(_messages.Message):
    """Configuration for the FHIR BigQuery and Cloud Storage schema. Determines
  how the server generates the schema.

  Enums:
    SchemaTypeValueValuesEnum: Specifies the output schema type. Schema type
      is required.

  Fields:
    recursiveStructureDepth: The depth for all recursive structures in the
      output analytics schema. For example, `concept` in the CodeSystem
      resource is a recursive structure; when the depth is 2, the CodeSystem
      table will have a column called `concept.concept` but not
      `concept.concept.concept`. If not specified or set to 0, the server will
      use the default value 2. The maximum depth allowed is 5.
    schemaType: Specifies the output schema type. Schema type is required.
  """

    class SchemaTypeValueValuesEnum(_messages.Enum):
        """Specifies the output schema type. Schema type is required.

    Values:
      SCHEMA_TYPE_UNSPECIFIED: No schema type specified. This type is
        unsupported.
      LOSSLESS: A data-driven schema generated from the fields present in the
        FHIR data being exported, with no additional simplification. This type
        cannot be used for streaming to BigQuery.
      ANALYTICS: Analytics schema defined by the FHIR community. See
        https://github.com/FHIR/sql-on-fhir/blob/master/sql-on-fhir.md.
        BigQuery only allows a maximum of 10,000 columns per table. Due to
        this limitation, the server will not generate schemas for fields of
        type `Resource`, which can hold any resource type. The affected fields
        are `Parameters.parameter.resource`, `Bundle.entry.resource`, and
        `Bundle.entry.response.outcome`. Analytics schema does not gracefully
        handle extensions with one or more occurrences, anaytics schema also
        does not handle contained resource.
      ANALYTICS_V2: Analytics V2, similar to schema defined by the FHIR
        community, with added support for extensions with one or more
        occurrences and contained resources in stringified JSON. Analytics V2
        uses more space in the destination table than Analytics V1. It is
        generally recommended to use Analytics V2 over Analytics.
    """
        SCHEMA_TYPE_UNSPECIFIED = 0
        LOSSLESS = 1
        ANALYTICS = 2
        ANALYTICS_V2 = 3
    recursiveStructureDepth = _messages.IntegerField(1)
    schemaType = _messages.EnumField('SchemaTypeValueValuesEnum', 2)
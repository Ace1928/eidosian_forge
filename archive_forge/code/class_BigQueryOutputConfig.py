from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class BigQueryOutputConfig(_messages.Message):
    """Configuration for the output that is specific to BigQuery when choosing
  a BigQuery dataset as the output destination.

  Enums:
    VariantValueValuesEnum: Schema variant that should be used when exporting
      data to BigQuery.

  Fields:
    variant: Schema variant that should be used when exporting data to
      BigQuery.
  """

    class VariantValueValuesEnum(_messages.Enum):
        """Schema variant that should be used when exporting data to BigQuery.

    Values:
      SCHEMA_VARIANT_UNSPECIFIED: (Input only) sentinel indicating that the
        API should choose automatically. On create, a value will be selected
        and persisted. Subsequent reads from the API will return a non-default
        value indicating what was selected.
      FIELDS_EXPANDED_AS_COLUMNS: Export data with all attributes and label
        keys of the span expanded as columns dynamically, and each span as a
        row. The attribute/label columns are assigned their associated values
        in that row. We recommend this mode of operation when the number of
        unique label keys are relatively small and stable, as it provides a
        simpler, more convenient query experience.
      CONDENSED_STATIC_JSON: Export data as condensed, json-formatted columns.
        Choose this mode when unique label keys have high cardinality and/or
        change frequently. This schema variant is more scalable and lacks the
        potential issue of reaching the maximum column limit in BigQuery. At
        the same time, this variant is more difficult to query compared with
        `FIELDS_EXPANDED_AS_COLUMNS` mode, often requiring the use of BigQuery
        JSON functions
        (https://cloud.google.com/bigquery/docs/reference/standard-
        sql/json_functions) in order to extract particular labels and use them
        in queries.
    """
        SCHEMA_VARIANT_UNSPECIFIED = 0
        FIELDS_EXPANDED_AS_COLUMNS = 1
        CONDENSED_STATIC_JSON = 2
    variant = _messages.EnumField('VariantValueValuesEnum', 1)
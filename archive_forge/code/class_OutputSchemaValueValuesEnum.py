from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OutputSchemaValueValuesEnum(_messages.Enum):
    """Schema used for writing the findings for Inspect jobs. This field is
    only used for Inspect and must be unspecified for Risk jobs. Columns are
    derived from the `Finding` object. If appending to an existing table, any
    columns from the predefined schema that are missing will be added. No
    columns in the existing table will be deleted. If unspecified, then all
    available columns will be used for a new table or an (existing) table with
    no schema, and no changes will be made to an existing table that has a
    schema. Only for use with external storage.

    Values:
      OUTPUT_SCHEMA_UNSPECIFIED: Unused.
      BASIC_COLUMNS: Basic schema including only `info_type`, `quote`,
        `certainty`, and `timestamp`.
      GCS_COLUMNS: Schema tailored to findings from scanning Cloud Storage.
      DATASTORE_COLUMNS: Schema tailored to findings from scanning Google
        Datastore.
      BIG_QUERY_COLUMNS: Schema tailored to findings from scanning Google
        BigQuery.
      ALL_COLUMNS: Schema containing all columns.
    """
    OUTPUT_SCHEMA_UNSPECIFIED = 0
    BASIC_COLUMNS = 1
    GCS_COLUMNS = 2
    DATASTORE_COLUMNS = 3
    BIG_QUERY_COLUMNS = 4
    ALL_COLUMNS = 5
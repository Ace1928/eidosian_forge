from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JobConfigurationQuery(_messages.Message):
    """A JobConfigurationQuery object.

  Messages:
    TableDefinitionsValue: [Optional] If querying an external data source
      outside of BigQuery, describes the data format, location and other
      properties of the data source. By defining these properties, the data
      source can then be queried as if it were a standard BigQuery table.

  Fields:
    allowLargeResults: If true, allows the query to produce arbitrarily large
      result tables at a slight cost in performance. Requires destinationTable
      to be set.
    createDisposition: [Optional] Specifies whether the job is allowed to
      create new tables. The following values are supported: CREATE_IF_NEEDED:
      If the table does not exist, BigQuery creates the table. CREATE_NEVER:
      The table must already exist. If it does not, a 'notFound' error is
      returned in the job result. The default value is CREATE_IF_NEEDED.
      Creation, truncation and append actions occur as one atomic update upon
      job completion.
    defaultDataset: [Optional] Specifies the default dataset to use for
      unqualified table names in the query.
    destinationTable: [Optional] Describes the table where the query results
      should be stored. If not present, a new table will be created to store
      the results.
    flattenResults: [Optional] Flattens all nested and repeated fields in the
      query results. The default value is true. allowLargeResults must be true
      if this is set to false.
    maximumBillingTier: [Optional] Limits the billing tier for this job.
      Queries that have resource usage beyond this tier will fail (without
      incurring a charge). If unspecified, this will be set to your project
      default.
    maximumBytesBilled: [Optional] Limits the bytes billed for this job.
      Queries that will have bytes billed beyond this limit will fail (without
      incurring a charge). If unspecified, this will be set to your project
      default.
    preserveNulls: [Deprecated] This property is deprecated.
    priority: [Optional] Specifies a priority for the query. Possible values
      include INTERACTIVE and BATCH. The default value is INTERACTIVE.
    query: [Required] BigQuery SQL query to execute.
    schemaUpdateOptions: [Experimental] Allows the schema of the desitination
      table to be updated as a side effect of the query job. Schema update
      options are supported in two cases: when writeDisposition is
      WRITE_APPEND; when writeDisposition is WRITE_TRUNCATE and the
      destination table is a partition of a table, specified by partition
      decorators. For normal tables, WRITE_TRUNCATE will always overwrite the
      schema. One or more of the following values are specified:
      ALLOW_FIELD_ADDITION: allow adding a nullable field to the schema.
      ALLOW_FIELD_RELAXATION: allow relaxing a required field in the original
      schema to nullable.
    tableDefinitions: [Optional] If querying an external data source outside
      of BigQuery, describes the data format, location and other properties of
      the data source. By defining these properties, the data source can then
      be queried as if it were a standard BigQuery table.
    useLegacySql: [Experimental] Specifies whether to use BigQuery's legacy
      SQL dialect for this query. The default value is true. If set to false,
      the query will use BigQuery's standard SQL:
      https://cloud.google.com/bigquery/sql-reference/ When useLegacySql is
      set to false, the values of allowLargeResults and flattenResults are
      ignored; query will be run as if allowLargeResults is true and
      flattenResults is false.
    useQueryCache: [Optional] Whether to look for the result in the query
      cache. The query cache is a best-effort cache that will be flushed
      whenever tables in the query are modified. Moreover, the query cache is
      only available when a query does not have a destination table specified.
      The default value is true.
    userDefinedFunctionResources: [Experimental] Describes user-defined
      function resources used in the query.
    writeDisposition: [Optional] Specifies the action that occurs if the
      destination table already exists. The following values are supported:
      WRITE_TRUNCATE: If the table already exists, BigQuery overwrites the
      table data. WRITE_APPEND: If the table already exists, BigQuery appends
      the data to the table. WRITE_EMPTY: If the table already exists and
      contains data, a 'duplicate' error is returned in the job result. The
      default value is WRITE_EMPTY. Each action is atomic and only occurs if
      BigQuery is able to complete the job successfully. Creation, truncation
      and append actions occur as one atomic update upon job completion.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class TableDefinitionsValue(_messages.Message):
        """[Optional] If querying an external data source outside of BigQuery,
    describes the data format, location and other properties of the data
    source. By defining these properties, the data source can then be queried
    as if it were a standard BigQuery table.

    Messages:
      AdditionalProperty: An additional property for a TableDefinitionsValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        TableDefinitionsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a TableDefinitionsValue object.

      Fields:
        key: Name of the additional property.
        value: A ExternalDataConfiguration attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('ExternalDataConfiguration', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    allowLargeResults = _messages.BooleanField(1)
    createDisposition = _messages.StringField(2)
    defaultDataset = _messages.MessageField('DatasetReference', 3)
    destinationTable = _messages.MessageField('TableReference', 4)
    flattenResults = _messages.BooleanField(5, default=True)
    maximumBillingTier = _messages.IntegerField(6, variant=_messages.Variant.INT32, default=1)
    maximumBytesBilled = _messages.IntegerField(7)
    preserveNulls = _messages.BooleanField(8)
    priority = _messages.StringField(9)
    query = _messages.StringField(10)
    schemaUpdateOptions = _messages.StringField(11, repeated=True)
    tableDefinitions = _messages.MessageField('TableDefinitionsValue', 12)
    useLegacySql = _messages.BooleanField(13)
    useQueryCache = _messages.BooleanField(14, default=True)
    userDefinedFunctionResources = _messages.MessageField('UserDefinedFunctionResource', 15, repeated=True)
    writeDisposition = _messages.StringField(16)
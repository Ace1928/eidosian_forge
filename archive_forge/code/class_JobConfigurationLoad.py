from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JobConfigurationLoad(_messages.Message):
    """A JobConfigurationLoad object.

  Fields:
    allowJaggedRows: [Optional] Accept rows that are missing trailing optional
      columns. The missing values are treated as nulls. If false, records with
      missing trailing columns are treated as bad records, and if there are
      too many bad records, an invalid error is returned in the job result.
      The default value is false. Only applicable to CSV, ignored for other
      formats.
    allowQuotedNewlines: Indicates if BigQuery should allow quoted data
      sections that contain newline characters in a CSV file. The default
      value is false.
    autodetect: [Experimental] Indicates if we should automatically infer the
      options and schema for CSV and JSON sources.
    createDisposition: [Optional] Specifies whether the job is allowed to
      create new tables. The following values are supported: CREATE_IF_NEEDED:
      If the table does not exist, BigQuery creates the table. CREATE_NEVER:
      The table must already exist. If it does not, a 'notFound' error is
      returned in the job result. The default value is CREATE_IF_NEEDED.
      Creation, truncation and append actions occur as one atomic update upon
      job completion.
    destinationTable: [Required] The destination table to load the data into.
    encoding: [Optional] The character encoding of the data. The supported
      values are UTF-8 or ISO-8859-1. The default value is UTF-8. BigQuery
      decodes the data after the raw, binary data has been split using the
      values of the quote and fieldDelimiter properties.
    fieldDelimiter: [Optional] The separator for fields in a CSV file. The
      separator can be any ISO-8859-1 single-byte character. To use a
      character in the range 128-255, you must encode the character as UTF8.
      BigQuery converts the string to ISO-8859-1 encoding, and then uses the
      first byte of the encoded string to split the data in its raw, binary
      state. BigQuery also supports the escape sequence "\\t" to specify a tab
      separator. The default value is a comma (',').
    ignoreUnknownValues: [Optional] Indicates if BigQuery should allow extra
      values that are not represented in the table schema. If true, the extra
      values are ignored. If false, records with extra columns are treated as
      bad records, and if there are too many bad records, an invalid error is
      returned in the job result. The default value is false. The sourceFormat
      property determines what BigQuery treats as an extra value: CSV:
      Trailing columns JSON: Named values that don't match any column names
    maxBadRecords: [Optional] The maximum number of bad records that BigQuery
      can ignore when running the job. If the number of bad records exceeds
      this value, an invalid error is returned in the job result. The default
      value is 0, which requires that all records are valid.
    projectionFields: [Experimental] If sourceFormat is set to
      "DATASTORE_BACKUP", indicates which entity properties to load into
      BigQuery from a Cloud Datastore backup. Property names are case
      sensitive and must be top-level properties. If no properties are
      specified, BigQuery loads all properties. If any named property isn't
      found in the Cloud Datastore backup, an invalid error is returned in the
      job result.
    quote: [Optional] The value that is used to quote data sections in a CSV
      file. BigQuery converts the string to ISO-8859-1 encoding, and then uses
      the first byte of the encoded string to split the data in its raw,
      binary state. The default value is a double-quote ('"'). If your data
      does not contain quoted sections, set the property value to an empty
      string. If your data contains quoted newline characters, you must also
      set the allowQuotedNewlines property to true.
    schema: [Optional] The schema for the destination table. The schema can be
      omitted if the destination table already exists, or if you're loading
      data from Google Cloud Datastore.
    schemaInline: [Deprecated] The inline schema. For CSV schemas, specify as
      "Field1:Type1[,Field2:Type2]*". For example, "foo:STRING, bar:INTEGER,
      baz:FLOAT".
    schemaInlineFormat: [Deprecated] The format of the schemaInline property.
    schemaUpdateOptions: [Experimental] Allows the schema of the desitination
      table to be updated as a side effect of the load job. Schema update
      options are supported in two cases: when writeDisposition is
      WRITE_APPEND; when writeDisposition is WRITE_TRUNCATE and the
      destination table is a partition of a table, specified by partition
      decorators. For normal tables, WRITE_TRUNCATE will always overwrite the
      schema. One or more of the following values are specified:
      ALLOW_FIELD_ADDITION: allow adding a nullable field to the schema.
      ALLOW_FIELD_RELAXATION: allow relaxing a required field in the original
      schema to nullable.
    skipLeadingRows: [Optional] The number of rows at the top of a CSV file
      that BigQuery will skip when loading the data. The default value is 0.
      This property is useful if you have header rows in the file that should
      be skipped.
    sourceFormat: [Optional] The format of the data files. For CSV files,
      specify "CSV". For datastore backups, specify "DATASTORE_BACKUP". For
      newline-delimited JSON, specify "NEWLINE_DELIMITED_JSON". For Avro,
      specify "AVRO". The default value is CSV.
    sourceUris: [Required] The fully-qualified URIs that point to your data in
      Google Cloud Storage. Each URI can contain one '*' wildcard character
      and it must come after the 'bucket' name.
    writeDisposition: [Optional] Specifies the action that occurs if the
      destination table already exists. The following values are supported:
      WRITE_TRUNCATE: If the table already exists, BigQuery overwrites the
      table data. WRITE_APPEND: If the table already exists, BigQuery appends
      the data to the table. WRITE_EMPTY: If the table already exists and
      contains data, a 'duplicate' error is returned in the job result. The
      default value is WRITE_APPEND. Each action is atomic and only occurs if
      BigQuery is able to complete the job successfully. Creation, truncation
      and append actions occur as one atomic update upon job completion.
  """
    allowJaggedRows = _messages.BooleanField(1)
    allowQuotedNewlines = _messages.BooleanField(2)
    autodetect = _messages.BooleanField(3)
    createDisposition = _messages.StringField(4)
    destinationTable = _messages.MessageField('TableReference', 5)
    encoding = _messages.StringField(6)
    fieldDelimiter = _messages.StringField(7)
    ignoreUnknownValues = _messages.BooleanField(8)
    maxBadRecords = _messages.IntegerField(9, variant=_messages.Variant.INT32)
    projectionFields = _messages.StringField(10, repeated=True)
    quote = _messages.StringField(11, default=u'"')
    schema = _messages.MessageField('TableSchema', 12)
    schemaInline = _messages.StringField(13)
    schemaInlineFormat = _messages.StringField(14)
    schemaUpdateOptions = _messages.StringField(15, repeated=True)
    skipLeadingRows = _messages.IntegerField(16, variant=_messages.Variant.INT32)
    sourceFormat = _messages.StringField(17)
    sourceUris = _messages.StringField(18, repeated=True)
    writeDisposition = _messages.StringField(19)
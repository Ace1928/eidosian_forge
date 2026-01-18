from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TableDataInsertAllRequest(_messages.Message):
    """A TableDataInsertAllRequest object.

  Messages:
    RowsValueListEntry: A RowsValueListEntry object.

  Fields:
    ignoreUnknownValues: [Optional] Accept rows that contain values that do
      not match the schema. The unknown values are ignored. Default is false,
      which treats unknown values as errors.
    kind: The resource type of the response.
    rows: The rows to insert.
    skipInvalidRows: [Optional] Insert all valid rows of a request, even if
      invalid rows exist. The default value is false, which causes the entire
      request to fail if any invalid rows exist.
    templateSuffix: [Experimental] If specified, treats the destination table
      as a base template, and inserts the rows into an instance table named
      "{destination}{templateSuffix}". BigQuery will manage creation of the
      instance table, using the schema of the base template table. See
      https://cloud.google.com/bigquery/streaming-data-into-bigquery#template-
      tables for considerations when working with templates tables.
  """

    class RowsValueListEntry(_messages.Message):
        """A RowsValueListEntry object.

    Fields:
      insertId: [Optional] A unique ID for each row. BigQuery uses this
        property to detect duplicate insertion requests on a best-effort
        basis.
      json: [Required] A JSON object that contains a row of data. The object's
        properties and values must match the destination table's schema.
    """
        insertId = _messages.StringField(1)
        json = _messages.MessageField('JsonObject', 2)
    ignoreUnknownValues = _messages.BooleanField(1)
    kind = _messages.StringField(2, default=u'bigquery#tableDataInsertAllRequest')
    rows = _messages.MessageField('RowsValueListEntry', 3, repeated=True)
    skipInvalidRows = _messages.BooleanField(4)
    templateSuffix = _messages.StringField(5)
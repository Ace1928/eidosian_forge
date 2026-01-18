from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryResult(_messages.Message):
    """Execution results of the query. The result is formatted as rows
  represented by BigQuery compatible [schema]. When pagination is necessary,
  it will contains the page token to retrieve the results of following pages.

  Messages:
    RowsValueListEntry: A RowsValueListEntry object.

  Fields:
    nextPageToken: Token to retrieve the next page of the results.
    rows: Each row hold a query result in the format of `Struct`.
    schema: Describes the format of the [rows].
    totalRows: Total rows of the whole query results.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class RowsValueListEntry(_messages.Message):
        """A RowsValueListEntry object.

    Messages:
      AdditionalProperty: An additional property for a RowsValueListEntry
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a RowsValueListEntry object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    nextPageToken = _messages.StringField(1)
    rows = _messages.MessageField('RowsValueListEntry', 2, repeated=True)
    schema = _messages.MessageField('TableSchema', 3)
    totalRows = _messages.IntegerField(4)
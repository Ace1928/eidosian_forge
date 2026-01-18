from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryResults(_messages.Message):
    """Results of a SQL query over logs. Next ID: 11

  Messages:
    RowsValueListEntry: A RowsValueListEntry object.

  Fields:
    executionDuration: The total execution duration of the query.
    nextPageToken: A token that can be sent as page_token to retrieve the next
      page. If this field is omitted, there are no subsequent pages.
    queryComplete: Whether the query has completed or not. If this is false,
      the rows, total_rows, and execution_time fields will not be populated.
      The client needs to poll on ReadQueryResults specifying the
      result_reference and wait for results.
    restrictionConflicts: Conflicts between the query and the restrictions
      that were requested. Any restrictions present here were ignored when
      executing the query.
    resultReference: An opaque string that can be used as a reference to this
      query result. This result reference can be used in the QueryData query
      to fetch this result up to 24 hours in the future.
    rows: Query result rows. The number of rows returned depends upon the page
      size requested. To get any additional rows, you can call
      ReadQueryResults and specify the result_reference and the page_token.The
      REST-based representation of this data leverages a series of JSON f,v
      objects for indicating fields and values.
    schema: The schema of the results. It shows the columns present in the
      output table. Present only when the query completes successfully.
    totalBytesProcessed: The total number of bytes processed for this query.
      If this query was a validate_only query, this is the number of bytes
      that would be processed if the query were run.
    totalRows: The total number of rows in the complete query result set,
      which can be more than the number of rows in this single page of
      results.
    totalSlotMs: The total slot-milliseconds consumed by this query. Populated
      only on a call to ReadQueryResults.
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
    executionDuration = _messages.StringField(1)
    nextPageToken = _messages.StringField(2)
    queryComplete = _messages.BooleanField(3)
    restrictionConflicts = _messages.MessageField('QueryRestrictionConflict', 4, repeated=True)
    resultReference = _messages.StringField(5)
    rows = _messages.MessageField('RowsValueListEntry', 6, repeated=True)
    schema = _messages.MessageField('TableSchema', 7)
    totalBytesProcessed = _messages.IntegerField(8)
    totalRows = _messages.IntegerField(9)
    totalSlotMs = _messages.IntegerField(10)
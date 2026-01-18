from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResultSet(_messages.Message):
    """Results from Read or ExecuteSql.

  Messages:
    RowsValueListEntry: Single entry in a RowsValue.

  Fields:
    metadata: Metadata about the result set, such as row type information.
    rows: Each element in `rows` is a row whose format is defined by
      metadata.row_type. The ith element in each row matches the ith field in
      metadata.row_type. Elements are encoded based on type as described here.
    stats: Query plan and execution statistics for the SQL statement that
      produced this result set. These can be requested by setting
      ExecuteSqlRequest.query_mode. DML statements always produce stats
      containing the number of rows modified, unless executed using the
      ExecuteSqlRequest.QueryMode.PLAN ExecuteSqlRequest.query_mode. Other
      fields may or may not be populated, based on the
      ExecuteSqlRequest.query_mode.
  """

    class RowsValueListEntry(_messages.Message):
        """Single entry in a RowsValue.

    Fields:
      entry: A extra_types.JsonValue attribute.
    """
        entry = _messages.MessageField('extra_types.JsonValue', 1, repeated=True)
    metadata = _messages.MessageField('ResultSetMetadata', 1)
    rows = _messages.MessageField('RowsValueListEntry', 2, repeated=True)
    stats = _messages.MessageField('ResultSetStats', 3)
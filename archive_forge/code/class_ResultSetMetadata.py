from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResultSetMetadata(_messages.Message):
    """Metadata about a ResultSet or PartialResultSet.

  Fields:
    rowType: Indicates the field names and types for the rows in the result
      set. For example, a SQL query like `"SELECT UserId, UserName FROM
      Users"` could return a `row_type` value like: "fields": [ { "name":
      "UserId", "type": { "code": "INT64" } }, { "name": "UserName", "type": {
      "code": "STRING" } }, ]
    transaction: If the read or SQL query began a transaction as a side-
      effect, the information about the new transaction is yielded here.
    undeclaredParameters: A SQL query can be parameterized. In PLAN mode,
      these parameters can be undeclared. This indicates the field names and
      types for those undeclared parameters in the SQL query. For example, a
      SQL query like `"SELECT * FROM Users where UserId = @userId and UserName
      = @userName "` could return a `undeclared_parameters` value like:
      "fields": [ { "name": "UserId", "type": { "code": "INT64" } }, { "name":
      "UserName", "type": { "code": "STRING" } }, ]
  """
    rowType = _messages.MessageField('StructType', 1)
    transaction = _messages.MessageField('Transaction', 2)
    undeclaredParameters = _messages.MessageField('StructType', 3)
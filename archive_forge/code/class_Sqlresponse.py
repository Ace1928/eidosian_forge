from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class Sqlresponse(_messages.Message):
    """Represents a response to an sql statement.

  Messages:
    RowsValueListEntry: Single entry in a RowsValue.

  Fields:
    columns: Columns in the table.
    kind: Type name: a template for an individual table.
    rows: The rows in the table. For each cell we print out whatever cell
      value (e.g., numeric, string) exists. Thus it is important that each
      cell contains only one value.
  """

    class RowsValueListEntry(_messages.Message):
        """Single entry in a RowsValue.

    Fields:
      entry: A extra_types.JsonValue attribute.
    """
        entry = _messages.MessageField('extra_types.JsonValue', 1, repeated=True)
    columns = _messages.StringField(1, repeated=True)
    kind = _messages.StringField(2, default=u'fusiontables#sqlresponse')
    rows = _messages.MessageField('RowsValueListEntry', 3, repeated=True)
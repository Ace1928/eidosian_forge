from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TablesValueListEntry(_messages.Message):
    """A TablesValueListEntry object.

    Fields:
      friendlyName: The user-friendly name for this table.
      id: An opaque ID of the table
      kind: The resource type.
      tableReference: A reference uniquely identifying the table.
      type: The type of table. Possible values are: TABLE, VIEW.
    """
    friendlyName = _messages.StringField(1)
    id = _messages.StringField(2)
    kind = _messages.StringField(3, default=u'bigquery#table')
    tableReference = _messages.MessageField('TableReference', 4)
    type = _messages.StringField(5)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MysqlDatabase(_messages.Message):
    """MySQL database.

  Fields:
    database: Database name.
    mysqlTables: Tables in the database.
  """
    database = _messages.StringField(1)
    mysqlTables = _messages.MessageField('MysqlTable', 2, repeated=True)
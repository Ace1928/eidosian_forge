from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MysqlObjectIdentifier(_messages.Message):
    """Mysql data source object identifier.

  Fields:
    database: Required. The database name.
    table: Required. The table name.
  """
    database = _messages.StringField(1)
    table = _messages.StringField(2)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourceSqlChange(_messages.Message):
    """Options to configure rule type SourceSqlChange. The rule is used to
  alter the sql code for database entities. The rule filter field can refer to
  one entity. The rule scope can be: StoredProcedure, Function, Trigger, View

  Fields:
    sqlCode: Required. Sql code for source (stored procedure, function,
      trigger or view)
  """
    sqlCode = _messages.StringField(1)
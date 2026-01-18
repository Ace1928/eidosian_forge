from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PostgresqlObjectIdentifier(_messages.Message):
    """PostgreSQL data source object identifier.

  Fields:
    schema: Required. The schema name.
    table: Required. The table name.
  """
    schema = _messages.StringField(1)
    table = _messages.StringField(2)
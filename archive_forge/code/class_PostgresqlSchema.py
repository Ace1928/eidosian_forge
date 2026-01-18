from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PostgresqlSchema(_messages.Message):
    """PostgreSQL schema.

  Fields:
    postgresqlTables: Tables in the schema.
    schema: Schema name.
  """
    postgresqlTables = _messages.MessageField('PostgresqlTable', 1, repeated=True)
    schema = _messages.StringField(2)
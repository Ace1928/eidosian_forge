from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SqlServerRdbms(_messages.Message):
    """SQLServer database structure.

  Fields:
    schemas: SQLServer schemas in the database server.
  """
    schemas = _messages.MessageField('SqlServerSchema', 1, repeated=True)
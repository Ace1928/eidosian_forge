from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OracleRdbms(_messages.Message):
    """Oracle database structure.

  Fields:
    oracleSchemas: Oracle schemas/databases in the database server.
  """
    oracleSchemas = _messages.MessageField('OracleSchema', 1, repeated=True)
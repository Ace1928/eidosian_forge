from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpecificStartPosition(_messages.Message):
    """CDC strategy to start replicating from a specific position in the
  source.

  Fields:
    mysqlLogPosition: MySQL specific log position to start replicating from.
    oracleScnPosition: Oracle SCN to start replicating from.
  """
    mysqlLogPosition = _messages.MessageField('MysqlLogPosition', 1)
    oracleScnPosition = _messages.MessageField('OracleScnPosition', 2)
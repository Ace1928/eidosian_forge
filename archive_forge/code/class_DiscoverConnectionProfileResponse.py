from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DiscoverConnectionProfileResponse(_messages.Message):
    """Response from a discover request.

  Fields:
    mysqlRdbms: Enriched MySQL RDBMS object.
    oracleRdbms: Enriched Oracle RDBMS object.
    postgresqlRdbms: Enriched PostgreSQL RDBMS object.
  """
    mysqlRdbms = _messages.MessageField('MysqlRdbms', 1)
    oracleRdbms = _messages.MessageField('OracleRdbms', 2)
    postgresqlRdbms = _messages.MessageField('PostgresqlRdbms', 3)
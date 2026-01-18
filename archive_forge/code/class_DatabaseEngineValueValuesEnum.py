from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatabaseEngineValueValuesEnum(_messages.Enum):
    """Required. The database engine used by the Cloud SQL instance that this
    connection configures.

    Values:
      DATABASE_ENGINE_UNKNOWN: An engine that is not currently supported by
        SDP.
      DATABASE_ENGINE_MYSQL: Cloud SQL for MySQL instance.
      DATABASE_ENGINE_POSTGRES: Cloud SQL for Postgres instance.
    """
    DATABASE_ENGINE_UNKNOWN = 0
    DATABASE_ENGINE_MYSQL = 1
    DATABASE_ENGINE_POSTGRES = 2
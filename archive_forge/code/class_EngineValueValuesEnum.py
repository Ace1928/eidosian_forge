from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EngineValueValuesEnum(_messages.Enum):
    """The database engine.

    Values:
      DATABASE_ENGINE_UNSPECIFIED: The source database engine of the migration
        job is unknown.
      MYSQL: The source engine is MySQL.
    """
    DATABASE_ENGINE_UNSPECIFIED = 0
    MYSQL = 1
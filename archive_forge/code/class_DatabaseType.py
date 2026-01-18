from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatabaseType(_messages.Message):
    """A message defining the database engine and provider.

  Enums:
    EngineValueValuesEnum: The database engine.
    ProviderValueValuesEnum: The database provider.

  Fields:
    engine: The database engine.
    provider: The database provider.
  """

    class EngineValueValuesEnum(_messages.Enum):
        """The database engine.

    Values:
      DATABASE_ENGINE_UNSPECIFIED: The source database engine of the migration
        job is unknown.
      MYSQL: The source engine is MySQL.
    """
        DATABASE_ENGINE_UNSPECIFIED = 0
        MYSQL = 1

    class ProviderValueValuesEnum(_messages.Enum):
        """The database provider.

    Values:
      DATABASE_PROVIDER_UNSPECIFIED: The database provider is unknown.
      CLOUDSQL: CloudSQL runs the database.
      RDS: RDS runs the database.
    """
        DATABASE_PROVIDER_UNSPECIFIED = 0
        CLOUDSQL = 1
        RDS = 2
    engine = _messages.EnumField('EngineValueValuesEnum', 1)
    provider = _messages.EnumField('ProviderValueValuesEnum', 2)
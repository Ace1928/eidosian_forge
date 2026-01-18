from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DiscoveryCloudSqlConditions(_messages.Message):
    """Requirements that must be true before a table is profiled for the first
  time.

  Enums:
    DatabaseEnginesValueListEntryValuesEnum:
    TypesValueListEntryValuesEnum:

  Fields:
    databaseEngines: Optional. Database engines that should be profiled.
      Optional. Defaults to ALL_SUPPORTED_DATABASE_ENGINES if unspecified.
    types: Data profiles will only be generated for the database resource
      types specified in this field. If not specified, defaults to
      [DATABASE_RESOURCE_TYPE_ALL_SUPPORTED_TYPES].
  """

    class DatabaseEnginesValueListEntryValuesEnum(_messages.Enum):
        """DatabaseEnginesValueListEntryValuesEnum enum type.

    Values:
      DATABASE_ENGINE_UNSPECIFIED: Unused.
      ALL_SUPPORTED_DATABASE_ENGINES: Include all supported database engines.
      MYSQL: MySql database.
      POSTGRES: PostGres database.
    """
        DATABASE_ENGINE_UNSPECIFIED = 0
        ALL_SUPPORTED_DATABASE_ENGINES = 1
        MYSQL = 2
        POSTGRES = 3

    class TypesValueListEntryValuesEnum(_messages.Enum):
        """TypesValueListEntryValuesEnum enum type.

    Values:
      DATABASE_RESOURCE_TYPE_UNSPECIFIED: Unused.
      DATABASE_RESOURCE_TYPE_ALL_SUPPORTED_TYPES: Includes database resource
        types that become supported at a later time.
      DATABASE_RESOURCE_TYPE_TABLE: Tables.
    """
        DATABASE_RESOURCE_TYPE_UNSPECIFIED = 0
        DATABASE_RESOURCE_TYPE_ALL_SUPPORTED_TYPES = 1
        DATABASE_RESOURCE_TYPE_TABLE = 2
    databaseEngines = _messages.EnumField('DatabaseEnginesValueListEntryValuesEnum', 1, repeated=True)
    types = _messages.EnumField('TypesValueListEntryValuesEnum', 2, repeated=True)
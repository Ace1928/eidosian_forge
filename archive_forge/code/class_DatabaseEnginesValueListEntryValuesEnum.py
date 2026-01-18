from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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
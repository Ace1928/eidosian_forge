from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SupportedDbVersionsValueListEntryValuesEnum(_messages.Enum):
    """SupportedDbVersionsValueListEntryValuesEnum enum type.

    Values:
      DATABASE_VERSION_UNSPECIFIED: This is an unknown database version.
      POSTGRES_13: DEPRECATED - The database version is Postgres 13.
      POSTGRES_14: The database version is Postgres 14.
      POSTGRES_15: The database version is Postgres 15.
    """
    DATABASE_VERSION_UNSPECIFIED = 0
    POSTGRES_13 = 1
    POSTGRES_14 = 2
    POSTGRES_15 = 3
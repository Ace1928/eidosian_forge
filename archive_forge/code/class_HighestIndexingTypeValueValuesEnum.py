from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HighestIndexingTypeValueValuesEnum(_messages.Enum):
    """Optional. Most important inclusion of this column.

    Values:
      INDEXING_TYPE_UNSPECIFIED: Unspecified.
      INDEXING_TYPE_NONE: Column not a part of an index.
      INDEXING_TYPE_NON_UNIQUE: Column Part of non unique index.
      INDEXING_TYPE_UNIQUE: Column part of unique index.
      INDEXING_TYPE_PRIMARY_KEY: Column part of the primary key.
    """
    INDEXING_TYPE_UNSPECIFIED = 0
    INDEXING_TYPE_NONE = 1
    INDEXING_TYPE_NON_UNIQUE = 2
    INDEXING_TYPE_UNIQUE = 3
    INDEXING_TYPE_PRIMARY_KEY = 4
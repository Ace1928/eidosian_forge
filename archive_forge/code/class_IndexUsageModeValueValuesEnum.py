from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IndexUsageModeValueValuesEnum(_messages.Enum):
    """Specifies the index usage mode for the query.

    Values:
      INDEX_USAGE_MODE_UNSPECIFIED: Index usage mode not specified.
      UNUSED: No vector indexes were used in the vector search query. See
        [`indexUnusedReasons`]
        (/bigquery/docs/reference/rest/v2/Job#IndexUnusedReason) for detailed
        reasons.
      PARTIALLY_USED: Part of the vector search query used vector indexes. See
        [`indexUnusedReasons`]
        (/bigquery/docs/reference/rest/v2/Job#IndexUnusedReason) for why other
        parts of the query did not use vector indexes.
      FULLY_USED: The entire vector search query used vector indexes.
    """
    INDEX_USAGE_MODE_UNSPECIFIED = 0
    UNUSED = 1
    PARTIALLY_USED = 2
    FULLY_USED = 3
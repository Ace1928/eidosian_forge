from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SearchStatistics(_messages.Message):
    """Statistics for a search query. Populated as part of JobStatistics2.

  Enums:
    IndexUsageModeValueValuesEnum: Specifies the index usage mode for the
      query.

  Fields:
    indexUnusedReasons: When `indexUsageMode` is `UNUSED` or `PARTIALLY_USED`,
      this field explains why indexes were not used in all or part of the
      search query. If `indexUsageMode` is `FULLY_USED`, this field is not
      populated.
    indexUsageMode: Specifies the index usage mode for the query.
  """

    class IndexUsageModeValueValuesEnum(_messages.Enum):
        """Specifies the index usage mode for the query.

    Values:
      INDEX_USAGE_MODE_UNSPECIFIED: Index usage mode not specified.
      UNUSED: No search indexes were used in the search query. See
        [`indexUnusedReasons`]
        (/bigquery/docs/reference/rest/v2/Job#IndexUnusedReason) for detailed
        reasons.
      PARTIALLY_USED: Part of the search query used search indexes. See
        [`indexUnusedReasons`]
        (/bigquery/docs/reference/rest/v2/Job#IndexUnusedReason) for why other
        parts of the query did not use search indexes.
      FULLY_USED: The entire search query used search indexes.
    """
        INDEX_USAGE_MODE_UNSPECIFIED = 0
        UNUSED = 1
        PARTIALLY_USED = 2
        FULLY_USED = 3
    indexUnusedReasons = _messages.MessageField('IndexUnusedReason', 1, repeated=True)
    indexUsageMode = _messages.EnumField('IndexUsageModeValueValuesEnum', 2)
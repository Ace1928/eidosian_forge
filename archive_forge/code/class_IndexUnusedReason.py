from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IndexUnusedReason(_messages.Message):
    """Reason about why no search index was used in the search query (or sub-
  query).

  Enums:
    CodeValueValuesEnum: Specifies the high-level reason for the scenario when
      no search index was used.

  Fields:
    baseTable: Specifies the base table involved in the reason that no search
      index was used.
    code: Specifies the high-level reason for the scenario when no search
      index was used.
    indexName: Specifies the name of the unused search index, if available.
    message: Free form human-readable reason for the scenario when no search
      index was used.
  """

    class CodeValueValuesEnum(_messages.Enum):
        """Specifies the high-level reason for the scenario when no search index
    was used.

    Values:
      CODE_UNSPECIFIED: Code not specified.
      INDEX_CONFIG_NOT_AVAILABLE: Indicates the search index configuration has
        not been created.
      PENDING_INDEX_CREATION: Indicates the search index creation has not been
        completed.
      BASE_TABLE_TRUNCATED: Indicates the base table has been truncated (rows
        have been removed from table with TRUNCATE TABLE statement) since the
        last time the search index was refreshed.
      INDEX_CONFIG_MODIFIED: Indicates the search index configuration has been
        changed since the last time the search index was refreshed.
      TIME_TRAVEL_QUERY: Indicates the search query accesses data at a
        timestamp before the last time the search index was refreshed.
      NO_PRUNING_POWER: Indicates the usage of search index will not
        contribute to any pruning improvement for the search function, e.g.
        when the search predicate is in a disjunction with other non-search
        predicates.
      UNINDEXED_SEARCH_FIELDS: Indicates the search index does not cover all
        fields in the search function.
      UNSUPPORTED_SEARCH_PATTERN: Indicates the search index does not support
        the given search query pattern.
      OPTIMIZED_WITH_MATERIALIZED_VIEW: Indicates the query has been optimized
        by using a materialized view.
      SECURED_BY_DATA_MASKING: Indicates the query has been secured by data
        masking, and thus search indexes are not applicable.
      MISMATCHED_TEXT_ANALYZER: Indicates that the search index and the search
        function call do not have the same text analyzer.
      BASE_TABLE_TOO_SMALL: Indicates the base table is too small (below a
        certain threshold). The index does not provide noticeable search
        performance gains when the base table is too small.
      BASE_TABLE_TOO_LARGE: Indicates that the total size of indexed base
        tables in your organization exceeds your region's limit and the index
        is not used in the query. To index larger base tables, you can use
        your own reservation for index-management jobs.
      ESTIMATED_PERFORMANCE_GAIN_TOO_LOW: Indicates that the estimated
        performance gain from using the search index is too low for the given
        search query.
      NOT_SUPPORTED_IN_STANDARD_EDITION: Indicates that search indexes can not
        be used for search query with STANDARD edition.
      INDEX_SUPPRESSED_BY_FUNCTION_OPTION: Indicates that an option in the
        search function that cannot make use of the index has been selected.
      QUERY_CACHE_HIT: Indicates that the query was cached, and thus the
        search index was not used.
      INTERNAL_ERROR: Indicates an internal error that causes the search index
        to be unused.
      OTHER_REASON: Indicates that the reason search indexes cannot be used in
        the query is not covered by any of the other IndexUnusedReason
        options.
    """
        CODE_UNSPECIFIED = 0
        INDEX_CONFIG_NOT_AVAILABLE = 1
        PENDING_INDEX_CREATION = 2
        BASE_TABLE_TRUNCATED = 3
        INDEX_CONFIG_MODIFIED = 4
        TIME_TRAVEL_QUERY = 5
        NO_PRUNING_POWER = 6
        UNINDEXED_SEARCH_FIELDS = 7
        UNSUPPORTED_SEARCH_PATTERN = 8
        OPTIMIZED_WITH_MATERIALIZED_VIEW = 9
        SECURED_BY_DATA_MASKING = 10
        MISMATCHED_TEXT_ANALYZER = 11
        BASE_TABLE_TOO_SMALL = 12
        BASE_TABLE_TOO_LARGE = 13
        ESTIMATED_PERFORMANCE_GAIN_TOO_LOW = 14
        NOT_SUPPORTED_IN_STANDARD_EDITION = 15
        INDEX_SUPPRESSED_BY_FUNCTION_OPTION = 16
        QUERY_CACHE_HIT = 17
        INTERNAL_ERROR = 18
        OTHER_REASON = 19
    baseTable = _messages.MessageField('TableReference', 1)
    code = _messages.EnumField('CodeValueValuesEnum', 2)
    indexName = _messages.StringField(3)
    message = _messages.StringField(4)
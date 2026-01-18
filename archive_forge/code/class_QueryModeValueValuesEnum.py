from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueryModeValueValuesEnum(_messages.Enum):
    """Used to control the amount of debugging information returned in
    ResultSetStats. If partition_token is set, query_mode can only be set to
    QueryMode.NORMAL.

    Values:
      NORMAL: The default mode. Only the statement results are returned.
      PLAN: This mode returns only the query plan, without any results or
        execution statistics information.
      PROFILE: This mode returns both the query plan and the execution
        statistics along with the results.
    """
    NORMAL = 0
    PLAN = 1
    PROFILE = 2
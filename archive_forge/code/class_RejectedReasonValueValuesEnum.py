from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RejectedReasonValueValuesEnum(_messages.Enum):
    """If present, specifies the reason why the materialized view was not
    chosen for the query.

    Values:
      REJECTED_REASON_UNSPECIFIED: Default unspecified value.
      NO_DATA: View has no cached data because it has not refreshed yet.
      COST: The estimated cost of the view is more expensive than another view
        or the base table. Note: The estimate cost might not match the billed
        cost.
      BASE_TABLE_TRUNCATED: View has no cached data because a base table is
        truncated.
      BASE_TABLE_DATA_CHANGE: View is invalidated because of a data change in
        one or more base tables. It could be any recent change if the
        [`max_staleness`](https://cloud.google.com/bigquery/docs/materialized-
        views-create#max_staleness) option is not set for the view, or
        otherwise any change outside of the staleness window.
      BASE_TABLE_PARTITION_EXPIRATION_CHANGE: View is invalidated because a
        base table's partition expiration has changed.
      BASE_TABLE_EXPIRED_PARTITION: View is invalidated because a base table's
        partition has expired.
      BASE_TABLE_INCOMPATIBLE_METADATA_CHANGE: View is invalidated because a
        base table has an incompatible metadata change.
      TIME_ZONE: View is invalidated because it was refreshed with a time zone
        other than that of the current job.
      OUT_OF_TIME_TRAVEL_WINDOW: View is outside the time travel window.
      BASE_TABLE_FINE_GRAINED_SECURITY_POLICY: View is inaccessible to the
        user because of a fine-grained security policy on one of its base
        tables.
      BASE_TABLE_TOO_STALE: One of the view's base tables is too stale. For
        example, the cached metadata of a biglake table needs to be updated.
    """
    REJECTED_REASON_UNSPECIFIED = 0
    NO_DATA = 1
    COST = 2
    BASE_TABLE_TRUNCATED = 3
    BASE_TABLE_DATA_CHANGE = 4
    BASE_TABLE_PARTITION_EXPIRATION_CHANGE = 5
    BASE_TABLE_EXPIRED_PARTITION = 6
    BASE_TABLE_INCOMPATIBLE_METADATA_CHANGE = 7
    TIME_ZONE = 8
    OUT_OF_TIME_TRAVEL_WINDOW = 9
    BASE_TABLE_FINE_GRAINED_SECURITY_POLICY = 10
    BASE_TABLE_TOO_STALE = 11
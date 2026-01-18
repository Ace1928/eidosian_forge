from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MaterializedViewDefinition(_messages.Message):
    """Definition and configuration of a materialized view.

  Fields:
    allowNonIncrementalDefinition: Optional. This option declares authors
      intention to construct a materialized view that will not be refreshed
      incrementally.
    enableRefresh: Optional. Enable automatic refresh of the materialized view
      when the base table is updated. The default value is "true".
    lastRefreshTime: Output only. The time when this materialized view was
      last refreshed, in milliseconds since the epoch.
    maxStaleness: [Optional] Max staleness of data that could be returned when
      materizlized view is queried (formatted as Google SQL Interval type).
    query: Required. A query whose results are persisted.
    refreshIntervalMs: Optional. The maximum frequency at which this
      materialized view will be refreshed. The default value is "1800000" (30
      minutes).
  """
    allowNonIncrementalDefinition = _messages.BooleanField(1)
    enableRefresh = _messages.BooleanField(2)
    lastRefreshTime = _messages.IntegerField(3)
    maxStaleness = _messages.BytesField(4)
    query = _messages.StringField(5)
    refreshIntervalMs = _messages.IntegerField(6)
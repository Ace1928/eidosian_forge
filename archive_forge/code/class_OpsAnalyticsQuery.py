from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OpsAnalyticsQuery(_messages.Message):
    """Preview: A query that produces an aggregated response and supporting
  data. This is a preview feature and may be subject to change before final
  release.

  Fields:
    sql: A SQL query to fetch time series, category series, or numeric series
      data.
  """
    sql = _messages.StringField(1)
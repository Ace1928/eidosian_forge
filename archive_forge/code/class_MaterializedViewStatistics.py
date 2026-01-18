from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MaterializedViewStatistics(_messages.Message):
    """Statistics of materialized views considered in a query job.

  Fields:
    materializedView: Materialized views considered for the query job. Only
      certain materialized views are used. For a detailed list, see the child
      message. If many materialized views are considered, then the list might
      be incomplete.
  """
    materializedView = _messages.MessageField('MaterializedView', 1, repeated=True)
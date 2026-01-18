from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LogsPanel(_messages.Message):
    """A widget that displays a stream of log.

  Fields:
    filter: A filter that chooses which log entries to return. See Advanced
      Logs Queries (https://cloud.google.com/logging/docs/view/advanced-
      queries). Only log entries that match the filter are returned. An empty
      filter matches all log entries.
    resourceNames: The names of logging resources to collect logs for.
      Currently only projects are supported. If empty, the widget will default
      to the host project.
  """
    filter = _messages.StringField(1)
    resourceNames = _messages.StringField(2, repeated=True)
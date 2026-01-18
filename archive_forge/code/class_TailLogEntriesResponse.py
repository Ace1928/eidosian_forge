from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TailLogEntriesResponse(_messages.Message):
    """Result returned from TailLogEntries.

  Fields:
    entries: A list of log entries. Each response in the stream will order
      entries with increasing values of LogEntry.timestamp. Ordering is not
      guaranteed between separate responses.
    suppressionInfo: If entries that otherwise would have been included in the
      session were not sent back to the client, counts of relevant entries
      omitted from the session with the reason that they were not included.
      There will be at most one of each reason per response. The counts
      represent the number of suppressed entries since the last streamed
      response.
  """
    entries = _messages.MessageField('LogEntry', 1, repeated=True)
    suppressionInfo = _messages.MessageField('SuppressionInfo', 2, repeated=True)
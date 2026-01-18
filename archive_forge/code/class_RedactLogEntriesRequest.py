from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RedactLogEntriesRequest(_messages.Message):
    """The parameters to RedactLogEntries.

  Fields:
    filter: Required. A filter specifying which log entries to redact. The
      filter must be no more than 20k characters. An empty filter matches all
      log entries.
    name: Required. Log bucket from which to redact log entries.For
      example:"projects/my-project/locations/global/buckets/my-source-bucket"
    reason: Required. The reason log entries need to be redacted. This field
      will be recorded in redacted log entries and should omit sensitive
      information. The reason is limited 1,024 characters.
  """
    filter = _messages.StringField(1)
    name = _messages.StringField(2)
    reason = _messages.StringField(3)
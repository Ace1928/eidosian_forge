from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CopyLogEntriesRequest(_messages.Message):
    """The parameters to CopyLogEntries.

  Fields:
    destination: Required. Destination to which to copy log entries.
    filter: Optional. A filter specifying which log entries to copy. The
      filter must be no more than 20k characters. An empty filter matches all
      log entries.
    name: Required. Log bucket from which to copy log entries.For
      example:"projects/my-project/locations/global/buckets/my-source-bucket"
  """
    destination = _messages.StringField(1)
    filter = _messages.StringField(2)
    name = _messages.StringField(3)
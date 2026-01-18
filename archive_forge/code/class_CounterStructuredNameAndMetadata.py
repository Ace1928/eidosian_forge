from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CounterStructuredNameAndMetadata(_messages.Message):
    """A single message which encapsulates structured name and metadata for a
  given counter.

  Fields:
    metadata: Metadata associated with a counter
    name: Structured name of the counter.
  """
    metadata = _messages.MessageField('CounterMetadata', 1)
    name = _messages.MessageField('CounterStructuredName', 2)
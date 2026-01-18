from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchCreateOccurrencesResponse(_messages.Message):
    """Response for creating occurrences in batch.

  Fields:
    occurrences: The occurrences that were created.
  """
    occurrences = _messages.MessageField('Occurrence', 1, repeated=True)
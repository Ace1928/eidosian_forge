from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchCreateOccurrencesRequest(_messages.Message):
    """Request to create occurrences in batch.

  Fields:
    occurrences: Required. The occurrences to create. Max allowed length is
      1000.
  """
    occurrences = _messages.MessageField('Occurrence', 1, repeated=True)
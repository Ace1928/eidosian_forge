from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudAiLargeModelsVisionRelativeTemporalPartition(_messages.Message):
    """For ease of use, assume that the start_offset is inclusive and the
  end_offset is exclusive. In mathematical terms, the partition would be
  written as [start_offset, end_offset).

  Fields:
    endOffset: End time offset of the partition.
    startOffset: Start time offset of the partition.
  """
    endOffset = _messages.StringField(1)
    startOffset = _messages.StringField(2)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StreamingComputationRanges(_messages.Message):
    """Describes full or partial data disk assignment information of the
  computation ranges.

  Fields:
    computationId: The ID of the computation.
    rangeAssignments: Data disk assignments for ranges from this computation.
  """
    computationId = _messages.StringField(1)
    rangeAssignments = _messages.MessageField('KeyRangeDataDiskAssignment', 2, repeated=True)
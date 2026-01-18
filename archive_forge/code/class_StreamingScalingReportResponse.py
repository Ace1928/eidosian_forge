from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StreamingScalingReportResponse(_messages.Message):
    """Contains per-user-worker streaming scaling recommendation from the
  backend.

  Fields:
    maximumThreadCount: Maximum thread count limit;
  """
    maximumThreadCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
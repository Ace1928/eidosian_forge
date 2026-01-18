from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProgressTimeseries(_messages.Message):
    """Information about the progress of some component of job execution.

  Fields:
    currentProgress: The current progress of the component, in the range
      [0,1].
    dataPoints: History of progress for the component. Points are sorted by
      time.
  """
    currentProgress = _messages.FloatField(1)
    dataPoints = _messages.MessageField('Point', 2, repeated=True)
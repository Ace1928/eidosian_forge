from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1CheckTrialEarlyStoppingStateResponse(_messages.Message):
    """The message will be placed in the response field of a completed
  google.longrunning.Operation associated with a CheckTrialEarlyStoppingState
  request.

  Fields:
    endTime: The time at which operation processing completed.
    shouldStop: True if the Trial should stop.
    startTime: The time at which the operation was started.
  """
    endTime = _messages.StringField(1)
    shouldStop = _messages.BooleanField(2)
    startTime = _messages.StringField(3)
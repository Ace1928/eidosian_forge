from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApplicationAttemptInfo(_messages.Message):
    """Specific attempt of an application.

  Fields:
    appSparkVersion: A string attribute.
    attemptId: A string attribute.
    completed: A boolean attribute.
    durationMillis: A string attribute.
    endTime: A string attribute.
    lastUpdated: A string attribute.
    sparkUser: A string attribute.
    startTime: A string attribute.
  """
    appSparkVersion = _messages.StringField(1)
    attemptId = _messages.StringField(2)
    completed = _messages.BooleanField(3)
    durationMillis = _messages.IntegerField(4)
    endTime = _messages.StringField(5)
    lastUpdated = _messages.StringField(6)
    sparkUser = _messages.StringField(7)
    startTime = _messages.StringField(8)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisGoogleDevtoolsCloudbuildV1TimeSpan(_messages.Message):
    """Start and end times for a build execution phase.

  Fields:
    endTime: End of time span.
    startTime: Start of time span.
  """
    endTime = _messages.StringField(1)
    startTime = _messages.StringField(2)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicyanalyzerV1ObservationPeriod(_messages.Message):
    """Represents data observation period.

  Fields:
    endTime: The observation end time. The time in this timestamp is always
      `07:00:00Z`.
    startTime: The observation start time. The time in this timestamp is
      always `07:00:00Z`.
  """
    endTime = _messages.StringField(1)
    startTime = _messages.StringField(2)
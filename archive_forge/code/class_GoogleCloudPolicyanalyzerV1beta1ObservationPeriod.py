from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicyanalyzerV1beta1ObservationPeriod(_messages.Message):
    """Represents data observation period.

  Fields:
    endTime: The observation end time.
    startTime: The observation start time.
  """
    endTime = _messages.StringField(1)
    startTime = _messages.StringField(2)
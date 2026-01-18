from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceOperationProgress(_messages.Message):
    """Encapsulates progress related information for a Cloud Spanner long
  running instance operations.

  Fields:
    endTime: If set, the time at which this operation failed or was completed
      successfully.
    progressPercent: Percent completion of the operation. Values are between 0
      and 100 inclusive.
    startTime: Time the request was received.
  """
    endTime = _messages.StringField(1)
    progressPercent = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    startTime = _messages.StringField(3)
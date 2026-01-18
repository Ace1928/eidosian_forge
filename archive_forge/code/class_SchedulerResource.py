from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SchedulerResource(_messages.Message):
    """Configuration for resources used by Airflow schedulers.

  Fields:
    count: Optional. The number of schedulers.
    cpu: Optional. CPU request and limit for a single Airflow scheduler
      replica.
    memoryGb: Optional. Memory (GB) request and limit for a single Airflow
      scheduler replica.
    storageGb: Optional. Storage (GB) request and limit for a single Airflow
      scheduler replica.
  """
    count = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    cpu = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    memoryGb = _messages.FloatField(3, variant=_messages.Variant.FLOAT)
    storageGb = _messages.FloatField(4, variant=_messages.Variant.FLOAT)
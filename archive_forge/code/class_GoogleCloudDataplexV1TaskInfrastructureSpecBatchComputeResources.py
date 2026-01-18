from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1TaskInfrastructureSpecBatchComputeResources(_messages.Message):
    """Batch compute resources associated with the task.

  Fields:
    executorsCount: Optional. Total number of job executors. Executor Count
      should be between 2 and 100. Default=2
    maxExecutorsCount: Optional. Max configurable executors. If
      max_executors_count > executors_count, then auto-scaling is enabled. Max
      Executor Count should be between 2 and 1000. Default=1000
  """
    executorsCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    maxExecutorsCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkerThreadScalingReport(_messages.Message):
    """Contains information about the thread scaling information of a worker.

  Fields:
    currentThreadCount: Current number of active threads in a worker.
  """
    currentThreadCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
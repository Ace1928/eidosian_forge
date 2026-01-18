from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TaskResourceUsage(_messages.Message):
    """TaskResourceUsage describes the resource usage of the task.

  Fields:
    coreHours: The CPU core hours the task consumes based on task requirement
      and run time.
  """
    coreHours = _messages.FloatField(1)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SchedulingOnInstanceStopAction(_messages.Message):
    """Defines the behaviour for instances with the instance_termination_action
  STOP.

  Fields:
    discardLocalSsd: If true, the contents of any attached Local SSD disks
      will be discarded else, the Local SSD data will be preserved when the
      instance is stopped at the end of the run duration/termination time.
  """
    discardLocalSsd = _messages.BooleanField(1)
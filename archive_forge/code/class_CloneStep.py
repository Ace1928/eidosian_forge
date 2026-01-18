from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloneStep(_messages.Message):
    """CloneStep holds information about the clone step progress.

  Fields:
    adaptingOs: Adapting OS step.
    endTime: The time the step has ended.
    instantiatingMigratedVm: Instantiating migrated VM step.
    preparingVmDisks: Preparing VM disks step.
    startTime: The time the step has started.
  """
    adaptingOs = _messages.MessageField('AdaptingOSStep', 1)
    endTime = _messages.StringField(2)
    instantiatingMigratedVm = _messages.MessageField('InstantiatingMigratedVMStep', 3)
    preparingVmDisks = _messages.MessageField('PreparingVMDisksStep', 4)
    startTime = _messages.StringField(5)
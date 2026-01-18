from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlInstancesRescheduleMaintenanceRequestBody(_messages.Message):
    """Reschedule options for maintenance windows.

  Fields:
    reschedule: Required. The type of the reschedule the user wants.
  """
    reschedule = _messages.MessageField('Reschedule', 1)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MaintenanceUpdatePolicy(_messages.Message):
    """MaintenanceUpdatePolicy defines the policy for system updates.

  Fields:
    denyMaintenancePeriods: Periods to deny maintenance. Currently limited to
      1.
    maintenanceWindows: Preferred windows to perform maintenance. Currently
      limited to 1.
  """
    denyMaintenancePeriods = _messages.MessageField('DenyMaintenancePeriod', 1, repeated=True)
    maintenanceWindows = _messages.MessageField('MaintenanceWindow', 2, repeated=True)
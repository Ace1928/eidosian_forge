from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareAutoRepairConfig(_messages.Message):
    """Specifies config to enable/disable auto repair. The cluster-health-
  controller is deployed only if Enabled is true.

  Fields:
    enabled: Whether auto repair is enabled.
  """
    enabled = _messages.BooleanField(1)
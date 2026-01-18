from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceGroupManagerAutoHealingPolicy(_messages.Message):
    """A InstanceGroupManagerAutoHealingPolicy object.

  Fields:
    healthCheck: The URL for the health check that signals autohealing.
    initialDelaySec: The initial delay is the number of seconds that a new VM
      takes to initialize and run its startup script. During a VM's initial
      delay period, the MIG ignores unsuccessful health checks because the VM
      might be in the startup process. This prevents the MIG from prematurely
      recreating a VM. If the health check receives a healthy response during
      the initial delay, it indicates that the startup process is complete and
      the VM is ready. The value of initial delay must be between 0 and 3600
      seconds. The default value is 0.
  """
    healthCheck = _messages.StringField(1)
    initialDelaySec = _messages.IntegerField(2, variant=_messages.Variant.INT32)
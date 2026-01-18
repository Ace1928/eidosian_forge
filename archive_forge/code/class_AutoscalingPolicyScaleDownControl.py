from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AutoscalingPolicyScaleDownControl(_messages.Message):
    """Configuration that allows for slower scale in so that even if Autoscaler
  recommends an abrupt scale in of a MIG, it will be throttled as specified by
  the parameters below.

  Fields:
    maxScaledDownReplicas: Maximum allowed number (or %) of VMs that can be
      deducted from the peak recommendation during the window autoscaler looks
      at when computing recommendations. Possibly all these VMs can be deleted
      at once so user service needs to be prepared to lose that many VMs in
      one step.
    timeWindowSec: How far back autoscaling looks when computing
      recommendations to include directives regarding slower scale in, as
      described above.
  """
    maxScaledDownReplicas = _messages.MessageField('FixedOrPercent', 1)
    timeWindowSec = _messages.IntegerField(2, variant=_messages.Variant.INT32)
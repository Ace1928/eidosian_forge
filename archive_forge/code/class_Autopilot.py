from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Autopilot(_messages.Message):
    """Autopilot is the configuration for Autopilot settings on the cluster.

  Fields:
    conversionStatus: Output only. ConversionStatus is the status of
      conversion between Autopilot and standard.
    enabled: Enable Autopilot
    workloadPolicyConfig: Workload policy configuration for Autopilot.
  """
    conversionStatus = _messages.MessageField('AutopilotConversionStatus', 1)
    enabled = _messages.BooleanField(2)
    workloadPolicyConfig = _messages.MessageField('WorkloadPolicyConfig', 3)
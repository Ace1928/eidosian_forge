from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProtectConfig(_messages.Message):
    """ProtectConfig defines the flags needed to enable/disable features for
  the Protect API.

  Enums:
    WorkloadVulnerabilityModeValueValuesEnum: Sets which mode to use for
      Protect workload vulnerability scanning feature.

  Fields:
    workloadConfig: WorkloadConfig defines which actions are enabled for a
      cluster's workload configurations.
    workloadVulnerabilityMode: Sets which mode to use for Protect workload
      vulnerability scanning feature.
  """

    class WorkloadVulnerabilityModeValueValuesEnum(_messages.Enum):
        """Sets which mode to use for Protect workload vulnerability scanning
    feature.

    Values:
      WORKLOAD_VULNERABILITY_MODE_UNSPECIFIED: Default value not specified.
      DISABLED: Disables Workload Vulnerability Scanning feature on the
        cluster.
      BASIC: Applies basic vulnerability scanning settings for cluster
        workloads.
    """
        WORKLOAD_VULNERABILITY_MODE_UNSPECIFIED = 0
        DISABLED = 1
        BASIC = 2
    workloadConfig = _messages.MessageField('WorkloadConfig', 1)
    workloadVulnerabilityMode = _messages.EnumField('WorkloadVulnerabilityModeValueValuesEnum', 2)
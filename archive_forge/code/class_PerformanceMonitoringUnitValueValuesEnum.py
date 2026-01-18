from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PerformanceMonitoringUnitValueValuesEnum(_messages.Enum):
    """Type of Performance Monitoring Unit (PMU) requested on node pool
    instances. If unset, PMU will not be available to the node.

    Values:
      PERFORMANCE_MONITORING_UNIT_UNSPECIFIED: PMU not enabled.
      ARCHITECTURAL: Architecturally defined non-LLC events.
      STANDARD: Most documented core/L2 events.
      ENHANCED: Most documented core/L2 and LLC events.
    """
    PERFORMANCE_MONITORING_UNIT_UNSPECIFIED = 0
    ARCHITECTURAL = 1
    STANDARD = 2
    ENHANCED = 3
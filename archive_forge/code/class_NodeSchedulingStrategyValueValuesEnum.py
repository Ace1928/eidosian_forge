from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NodeSchedulingStrategyValueValuesEnum(_messages.Enum):
    """Defines behaviour of k8s scheduler.

    Values:
      STRATEGY_UNSPECIFIED: Use default scheduling strategy.
      PRIORITIZE_LEAST_UTILIZED: Least utilized nodes will be prioritized by
        k8s scheduler.
      PRIORITIZE_MEDIUM_UTILIZED: Nodes with medium utilization will be
        prioritized by k8s scheduler. This option improves interoperability of
        scheduler with cluster autoscaler.
    """
    STRATEGY_UNSPECIFIED = 0
    PRIORITIZE_LEAST_UTILIZED = 1
    PRIORITIZE_MEDIUM_UTILIZED = 2
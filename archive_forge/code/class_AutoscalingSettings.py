from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import dataclasses
from typing import Any, Dict, List, Union
from googlecloudsdk.core import exceptions
@dataclasses.dataclass(frozen=True)
class AutoscalingSettings:
    """Represents the autoscaling settings for a private-cloud cluster.

  Uses None for empty settings.

  Attributes:
    min_cluster_node_count: The minimum number of nodes in the cluster.
    max_cluster_node_count: The maximum number of nodes in the cluster.
    cool_down_period: The cool down period for autoscaling.
    autoscaling_policies: The autoscaling policies for each node type.
  """
    min_cluster_node_count: int
    max_cluster_node_count: int
    cool_down_period: str
    autoscaling_policies: Dict[str, AutoscalingPolicy]
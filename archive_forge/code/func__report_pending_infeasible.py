import copy
import logging
import math
import operator
import os
import queue
import subprocess
import threading
import time
from collections import Counter, defaultdict, namedtuple
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Union
import yaml
import ray
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.event_summarizer import EventSummarizer
from ray.autoscaler._private.legacy_info_string import legacy_log_info_string
from ray.autoscaler._private.load_metrics import LoadMetrics
from ray.autoscaler._private.local.node_provider import (
from ray.autoscaler._private.node_launcher import BaseNodeLauncher, NodeLauncher
from ray.autoscaler._private.node_provider_availability_tracker import (
from ray.autoscaler._private.node_tracker import NodeTracker
from ray.autoscaler._private.prom_metrics import AutoscalerPrometheusMetrics
from ray.autoscaler._private.providers import _get_node_provider
from ray.autoscaler._private.resource_demand_scheduler import (
from ray.autoscaler._private.updater import NodeUpdaterThread
from ray.autoscaler._private.util import (
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
from ray.exceptions import RpcError
def _report_pending_infeasible(self, unfulfilled: List[ResourceDict]):
    """Emit event messages for infeasible or unschedulable tasks.

        This adds messages to the event summarizer for warning on infeasible
        or "cluster full" resource requests.

        Args:
            unfulfilled: List of resource demands that would be unfulfilled
                even after full scale-up.
        """
    assert self.resource_demand_scheduler
    pending = []
    infeasible = []
    for bundle in unfulfilled:
        placement_group = any(('_group_' in k or k == 'bundle' for k in bundle))
        if placement_group:
            continue
        if self.resource_demand_scheduler.is_feasible(bundle):
            pending.append(bundle)
        else:
            infeasible.append(bundle)
    if pending:
        if self.load_metrics.cluster_full_of_actors_detected:
            for request in pending:
                self.event_summarizer.add_once_per_interval('Warning: The following resource request cannot be scheduled right now: {}. This is likely due to all cluster resources being claimed by actors. Consider creating fewer actors or adding more nodes to this Ray cluster.'.format(request), key='pending_{}'.format(sorted(request.items())), interval_s=30)
    if infeasible:
        for request in infeasible:
            self.event_summarizer.add_once_per_interval('Error: No available node types can fulfill resource request {}. Add suitable node types to this cluster to resolve this issue.'.format(request), key='infeasible_{}'.format(sorted(request.items())), interval_s=30)
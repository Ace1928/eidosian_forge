import collections
import copy
import logging
import os
from abc import abstractmethod
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple
import ray
from ray._private.gcs_utils import PlacementGroupTableData
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.loader import load_function_or_class
from ray.autoscaler._private.node_provider_availability_tracker import (
from ray.autoscaler._private.util import (
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
from ray.core.generated.common_pb2 import PlacementStrategy
def _default_utilization_scorer(node_resources: ResourceDict, resources: List[ResourceDict], node_type: str, *, node_availability_summary: NodeAvailabilitySummary):
    return _resource_based_utilization_scorer(node_resources, resources, node_availability_summary=node_availability_summary)
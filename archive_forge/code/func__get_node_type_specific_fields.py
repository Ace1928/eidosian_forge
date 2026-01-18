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
def _get_node_type_specific_fields(self, node_id: str, fields_key: str) -> Any:
    assert self.provider
    fields = self.config[fields_key]
    node_tags = self.provider.node_tags(node_id)
    if TAG_RAY_USER_NODE_TYPE in node_tags:
        node_type = node_tags[TAG_RAY_USER_NODE_TYPE]
        if node_type not in self.available_node_types:
            raise ValueError(f'Unknown node type tag: {node_type}.')
        node_specific_config = self.available_node_types[node_type]
        if fields_key in node_specific_config:
            fields = node_specific_config[fields_key]
    return fields
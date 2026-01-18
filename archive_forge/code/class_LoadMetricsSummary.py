import collections
import copy
import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from numbers import Number, Real
from typing import Any, Dict, List, Optional, Tuple, Union
import ray
import ray._private.services as services
from ray._private.utils import (
from ray.autoscaler._private import constants
from ray.autoscaler._private.cli_logger import cli_logger
from ray.autoscaler._private.docker import validate_docker_config
from ray.autoscaler._private.local.config import prepare_local
from ray.autoscaler._private.providers import _get_default_config
from ray.autoscaler.tags import NODE_TYPE_LEGACY_HEAD, NODE_TYPE_LEGACY_WORKER
@dataclass
class LoadMetricsSummary:
    usage: Usage
    resource_demand: List[DictCount]
    pg_demand: List[DictCount]
    request_demand: List[DictCount]
    node_types: List[DictCount]
    head_ip: Optional[NodeIP] = None
    usage_by_node: Optional[Dict[str, Usage]] = None
    node_type_mapping: Optional[Dict[str, str]] = None
    idle_time_map: Optional[Dict[str, int]] = None
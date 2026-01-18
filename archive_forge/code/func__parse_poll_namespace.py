import asyncio
import logging
import os
import random
from asyncio.events import AbstractEventLoop
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, DefaultDict, Dict, Optional, Set, Tuple, Union
import ray
from ray._private.utils import get_or_create_event_loop
from ray.serve._private.common import ReplicaName
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.utils import format_actor_name
from ray.serve.generated.serve_pb2 import ActorNameList
from ray.serve.generated.serve_pb2 import EndpointInfo as EndpointInfoProto
from ray.serve.generated.serve_pb2 import EndpointSet, LongPollRequest, LongPollResult
from ray.serve.generated.serve_pb2 import UpdatedObject as UpdatedObjectProto
from ray.util import metrics
def _parse_poll_namespace(self, name: str):
    if name == LongPollNamespace.ROUTE_TABLE.name:
        return LongPollNamespace.ROUTE_TABLE
    elif name == LongPollNamespace.RUNNING_REPLICAS.name:
        return LongPollNamespace.RUNNING_REPLICAS
    else:
        return name
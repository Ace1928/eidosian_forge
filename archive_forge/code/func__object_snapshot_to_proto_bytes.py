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
def _object_snapshot_to_proto_bytes(self, key: KeyType, object_snapshot: Any) -> bytes:
    if key == LongPollNamespace.ROUTE_TABLE:
        xlang_endpoints = {str(endpoint_tag): EndpointInfoProto(route=endpoint_info.route) for endpoint_tag, endpoint_info in object_snapshot.items()}
        return EndpointSet(endpoints=xlang_endpoints).SerializeToString()
    elif isinstance(key, tuple) and key[0] == LongPollNamespace.RUNNING_REPLICAS:
        actor_name_list = [f'{ReplicaName.prefix}{format_actor_name(replica_info.replica_tag)}' for replica_info in object_snapshot]
        return ActorNameList(names=actor_name_list).SerializeToString()
    else:
        return str.encode(str(object_snapshot))
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
def _listen_result_to_proto_bytes(self, keys_to_updated_objects: Dict[KeyType, UpdatedObject]) -> bytes:
    xlang_keys_to_updated_objects = {self._build_xlang_key(key): UpdatedObjectProto(snapshot_id=updated_object.snapshot_id, object_snapshot=self._object_snapshot_to_proto_bytes(key, updated_object.object_snapshot)) for key, updated_object in keys_to_updated_objects.items()}
    data = {'updated_objects': xlang_keys_to_updated_objects}
    proto = LongPollResult(**data)
    return proto.SerializeToString()
import asyncio
from collections import deque
import logging
import random
from typing import Tuple, List
import grpc
from ray._private.utils import get_or_create_event_loop
import ray._private.gcs_utils as gcs_utils
import ray._private.logging_utils as logging_utils
from ray.core.generated.gcs_pb2 import ErrorTableData
from ray.core.generated import dependency_pb2
from ray.core.generated import gcs_service_pb2_grpc
from ray.core.generated import gcs_service_pb2
from ray.core.generated import common_pb2
from ray.core.generated import pubsub_pb2
@staticmethod
def _pop_actors(queue, batch_size=100):
    if len(queue) == 0:
        return []
    popped = 0
    msgs = []
    while len(queue) > 0 and popped < batch_size:
        msg = queue.popleft()
        msgs.append((msg.key_id, msg.actor_message))
        popped += 1
    return msgs
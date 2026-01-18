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
def _unsubscribe_request(self, channels):
    req = gcs_service_pb2.GcsSubscriberCommandBatchRequest(subscriber_id=self._subscriber_id, sender_id=self._worker_id, commands=[])
    for channel in channels:
        req.commands.append(pubsub_pb2.Command(channel_type=channel, unsubscribe_message={}))
    return req
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
def _create_log_request(log_json: dict):
    job_id = log_json.get('job')
    return gcs_service_pb2.GcsPublishRequest(pub_messages=[pubsub_pb2.PubMessage(channel_type=pubsub_pb2.RAY_LOG_CHANNEL, key_id=job_id.encode() if job_id else None, log_batch_message=logging_utils.log_batch_dict_to_proto(log_json))])
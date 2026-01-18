import math
import logging
import queue
import threading
import warnings
import grpc
from collections import OrderedDict
from typing import Any, Callable, Dict, TYPE_CHECKING, Optional, Union
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray.util.client.common import (
from ray.util.debug import log_once
def _acknowledge(self, req_id: int) -> None:
    """
        Puts an acknowledge request on the request queue periodically.
        Lock should be held before calling this. Used when an async or
        blocking response is received.
        """
    if not self.client_worker._reconnect_enabled:
        return
    assert self.lock.locked()
    self._acknowledge_counter += 1
    if self._acknowledge_counter % ACKNOWLEDGE_BATCH_SIZE == 0:
        self.request_queue.put(ray_client_pb2.DataRequest(acknowledge=ray_client_pb2.AcknowledgeRequest(req_id=req_id)))
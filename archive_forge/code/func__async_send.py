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
def _async_send(self, req: ray_client_pb2.DataRequest, callback: Optional[ResponseCallable]=None) -> None:
    with self.lock:
        self._check_shutdown()
        req_id = self._next_id()
        req.req_id = req_id
        self.asyncio_waiting_data[req_id] = callback
        self.outstanding_requests[req_id] = req
        self.request_queue.put(req)
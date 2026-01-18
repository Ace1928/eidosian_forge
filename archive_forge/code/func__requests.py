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
def _requests(self):
    while True:
        req = self.request_queue.get()
        if req is None:
            return
        req_type = req.WhichOneof('type')
        if req_type == 'put':
            yield from chunk_put(req)
        elif req_type == 'task':
            yield from chunk_task(req)
        else:
            yield req
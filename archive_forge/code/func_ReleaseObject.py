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
def ReleaseObject(self, request: ray_client_pb2.ReleaseRequest, context=None) -> None:
    datareq = ray_client_pb2.DataRequest(release=request)
    self._async_send(datareq)
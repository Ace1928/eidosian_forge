import sys
import logging
import queue
import threading
import time
import grpc
from typing import TYPE_CHECKING
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray.util.debug import log_once
def _start_logthread(self) -> threading.Thread:
    return threading.Thread(target=self._log_main, args=(), daemon=True)
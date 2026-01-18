import inspect
import logging
import os
import pickle
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import grpc
import ray._raylet as raylet
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray._private import ray_constants
from ray._private.inspect_util import (
from ray._private.signature import extract_signature, get_signature
from ray._private.utils import check_oversized_function
from ray.util.client import ray
from ray.util.client.options import validate_options
@dataclass
class ClientServerHandle:
    """Holds the handles to the registered gRPC servicers and their server."""
    task_servicer: ray_client_pb2_grpc.RayletDriverServicer
    data_servicer: ray_client_pb2_grpc.RayletDataStreamerServicer
    logs_servicer: ray_client_pb2_grpc.RayletLogStreamerServicer
    grpc_server: grpc.Server

    def stop(self, grace: int) -> None:
        self.grpc_server.stop(grace)
        self.data_servicer.stopped.set()

    def __getattr__(self, attr):
        return getattr(self.grpc_server, attr)
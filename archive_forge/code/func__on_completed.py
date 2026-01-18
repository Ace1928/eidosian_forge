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
def _on_completed(self, py_callback: Callable[[Any], None]) -> None:
    """Register a callback that will be called after Object is ready.
        If the ObjectRef is already ready, the callback will be called soon.
        The callback should take the result as the only argument. The result
        can be an exception object in case of task error.
        """

    def deserialize_obj(resp: Union[ray_client_pb2.DataResponse, Exception]) -> None:
        from ray.util.client.client_pickler import loads_from_server
        if isinstance(resp, Exception):
            data = resp
        elif isinstance(resp, bytearray):
            data = loads_from_server(resp)
        else:
            obj = resp.get
            data = None
            if not obj.valid:
                data = loads_from_server(resp.get.error)
            else:
                data = loads_from_server(resp.get.data)
        py_callback(data)
    self._worker.register_callback(self, deserialize_obj)
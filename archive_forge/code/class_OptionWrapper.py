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
class OptionWrapper:

    def __init__(self, stub: ClientStub, options: Optional[Dict[str, Any]]):
        self._remote_stub = stub
        self._options = validate_options(options)

    def remote(self, *args, **kwargs):
        self._remote_stub._signature.bind(*args, **kwargs)
        return return_refs(ray.call_remote(self, *args, **kwargs))

    def __getattr__(self, key):
        return getattr(self._remote_stub, key)

    def _prepare_client_task(self):
        task = self._remote_stub._prepare_client_task()
        set_task_options(task, self._options)
        return task

    def _num_returns(self) -> int:
        if self._options:
            num = self._options.get('num_returns')
            if num is not None:
                return num
        return self._remote_stub._num_returns()
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
class ActorOptionWrapper(OptionWrapper):

    def remote(self, *args, **kwargs):
        self._remote_stub._init_signature.bind(*args, **kwargs)
        futures = ray.call_remote(self, *args, **kwargs)
        assert len(futures) == 1
        actor_class = None
        if isinstance(self._remote_stub, ClientActorClass):
            actor_class = self._remote_stub
        return ClientActorHandle(ClientActorRef(futures[0]), actor_class=actor_class)
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
class ClientRemoteMethod(ClientStub):
    """A stub for a method on a remote actor.

    Can be annotated with execution options.

    Args:
        actor_handle: A reference to the ClientActorHandle that generated
          this method and will have this method called upon it.
        method_name: The name of this method
    """

    def __init__(self, actor_handle: ClientActorHandle, method_name: str, num_returns: int, signature: inspect.Signature):
        self._actor_handle = actor_handle
        self._method_name = method_name
        self._method_num_returns = num_returns
        self._signature = signature

    def __call__(self, *args, **kwargs):
        raise TypeError(f"Actor methods cannot be called directly. Instead of running 'object.{self._method_name}()', try 'object.{self._method_name}.remote()'.")

    def remote(self, *args, **kwargs):
        self._signature.bind(*args, **kwargs)
        return return_refs(ray.call_remote(self, *args, **kwargs))

    def __repr__(self):
        return 'ClientRemoteMethod(%s, %s, %s)' % (self._method_name, self._actor_handle, self._method_num_returns)

    def options(self, **kwargs):
        return OptionWrapper(self, kwargs)

    def _remote(self, args=None, kwargs=None, **option_args):
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        return self.options(**option_args).remote(*args, **kwargs)

    def _prepare_client_task(self) -> ray_client_pb2.ClientTask:
        task = ray_client_pb2.ClientTask()
        task.type = ray_client_pb2.ClientTask.METHOD
        task.name = self._method_name
        task.payload_id = self._actor_handle.actor_ref.id
        return task

    def _num_returns(self) -> int:
        return self._method_num_returns
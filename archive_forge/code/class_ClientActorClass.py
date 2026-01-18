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
class ClientActorClass(ClientStub):
    """A stub created on the Ray Client to represent an actor class.

    It is wrapped by ray.remote and can be executed on the cluster.

    Args:
        actor_cls: The actual class to execute remotely
        _name: The original name of the class
        _ref: The ClientObjectRef of the pickled `actor_cls`
    """

    def __init__(self, actor_cls, options=None):
        self.actor_cls = actor_cls
        self._lock = threading.Lock()
        self._name = actor_cls.__name__
        self._init_signature = inspect.Signature(parameters=extract_signature(actor_cls.__init__, ignore_first=True))
        self._ref = None
        self._client_side_ref = ClientSideRefID.generate_id()
        self._options = validate_options(options)

    def __call__(self, *args, **kwargs):
        raise TypeError(f'Remote actor cannot be instantiated directly. Use {self._name}.remote() instead')

    def _ensure_ref(self):
        with self._lock:
            if self._ref is None:
                self._ref = InProgressSentinel()
                data = ray.worker._dumps_from_client(self.actor_cls)
                check_oversized_function(data, self._name, 'actor', None)
                self._ref = ray.worker._put_pickled(data, client_ref_id=self._client_side_ref.id)

    def remote(self, *args, **kwargs) -> 'ClientActorHandle':
        self._init_signature.bind(*args, **kwargs)
        futures = ray.call_remote(self, *args, **kwargs)
        assert len(futures) == 1
        return ClientActorHandle(ClientActorRef(futures[0]), actor_class=self)

    def options(self, **kwargs):
        return ActorOptionWrapper(self, kwargs)

    def _remote(self, args=None, kwargs=None, **option_args):
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        return self.options(**option_args).remote(*args, **kwargs)

    def __repr__(self):
        return 'ClientActorClass(%s, %s)' % (self._name, self._ref)

    def __getattr__(self, key):
        if key not in self.__dict__:
            raise AttributeError('Not a class attribute')
        raise NotImplementedError('static methods')

    def _prepare_client_task(self) -> ray_client_pb2.ClientTask:
        self._ensure_ref()
        task = ray_client_pb2.ClientTask()
        task.type = ray_client_pb2.ClientTask.ACTOR
        task.name = self._name
        task.payload_id = self._ref.id
        set_task_options(task, self._options, 'baseline_options')
        return task

    @staticmethod
    def _num_returns() -> int:
        return 1
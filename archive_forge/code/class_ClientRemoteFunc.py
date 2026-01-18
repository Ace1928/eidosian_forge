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
class ClientRemoteFunc(ClientStub):
    """A stub created on the Ray Client to represent a remote
    function that can be exectued on the cluster.

    This class is allowed to be passed around between remote functions.

    Args:
        _func: The actual function to execute remotely
        _name: The original name of the function
        _ref: The ClientObjectRef of the pickled code of the function, _func
    """

    def __init__(self, f, options=None):
        self._lock = threading.Lock()
        self._func = f
        self._name = f.__name__
        self._signature = get_signature(f)
        self._ref = None
        self._client_side_ref = ClientSideRefID.generate_id()
        self._options = validate_options(options)

    def __call__(self, *args, **kwargs):
        raise TypeError(f'Remote function cannot be called directly. Use {self._name}.remote method instead')

    def remote(self, *args, **kwargs):
        self._signature.bind(*args, **kwargs)
        return return_refs(ray.call_remote(self, *args, **kwargs))

    def options(self, **kwargs):
        return OptionWrapper(self, kwargs)

    def _remote(self, args=None, kwargs=None, **option_args):
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        return self.options(**option_args).remote(*args, **kwargs)

    def __repr__(self):
        return 'ClientRemoteFunc(%s, %s)' % (self._name, self._ref)

    def _ensure_ref(self):
        with self._lock:
            if self._ref is None:
                self._ref = InProgressSentinel()
                data = ray.worker._dumps_from_client(self._func)
                check_oversized_function(data, self._name, 'remote function', None)
                self._ref = ray.worker._put_pickled(data, client_ref_id=self._client_side_ref.id)

    def _prepare_client_task(self) -> ray_client_pb2.ClientTask:
        self._ensure_ref()
        task = ray_client_pb2.ClientTask()
        task.type = ray_client_pb2.ClientTask.FUNCTION
        task.name = self._name
        task.payload_id = self._ref.id
        set_task_options(task, self._options, 'baseline_options')
        return task

    def _num_returns(self) -> int:
        if not self._options:
            return None
        return self._options.get('num_returns')
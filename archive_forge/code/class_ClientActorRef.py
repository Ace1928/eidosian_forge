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
class ClientActorRef(raylet.ActorID):

    def __init__(self, id: Union[bytes, Future]):
        self._mutex = threading.Lock()
        self._worker = ray.get_context().client_worker
        if isinstance(id, bytes):
            self._set_id(id)
            self._id_future = None
        elif isinstance(id, Future):
            self._id_future = id
        else:
            raise TypeError('Unexpected type for id {}'.format(id))

    def __del__(self):
        if self._worker is not None and self._worker.is_connected():
            try:
                if not self.is_nil():
                    self._worker.call_release(self.id)
            except Exception:
                logger.debug('Exception from actor creation is ignored in destructor. To receive this exception in application code, call a method on the actor reference before its destructor is run.')

    def binary(self):
        self._wait_for_id()
        return super().binary()

    def hex(self):
        self._wait_for_id()
        return super().hex()

    def is_nil(self):
        self._wait_for_id()
        return super().is_nil()

    def __hash__(self):
        self._wait_for_id()
        return hash(self.id)

    @property
    def id(self):
        return self.binary()

    def _set_id(self, id):
        super()._set_id(id)
        self._worker.call_retain(id)

    def _wait_for_id(self, timeout=None):
        if self._id_future:
            with self._mutex:
                if self._id_future:
                    self._set_id(self._id_future.result(timeout=timeout))
                    self._id_future = None
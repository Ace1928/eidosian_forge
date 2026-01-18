import base64
import functools
import gc
import inspect
import json
import logging
import math
import pickle
import queue
import threading
import time
from collections import defaultdict
from concurrent import futures
from typing import Any, Callable, Dict, List, Optional, Set, Union
import grpc
import ray
import ray._private.state
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray import cloudpickle
from ray._private import ray_constants
from ray._private.client_mode_hook import disable_client_hook
from ray._raylet import GcsClient
from ray._private.ray_constants import env_integer
from ray._private.ray_logging import setup_logger
from ray._private.services import canonicalize_bootstrap_address_or_die
from ray._private.tls_utils import add_port_to_grpc_server
from ray.job_config import JobConfig
from ray.util.client.common import (
from ray.util.client.server.dataservicer import DataServicer
from ray.util.client.server.logservicer import LogstreamServicer
from ray.util.client.server.proxier import serve_proxier
from ray.util.client.server.server_pickler import dumps_from_server, loads_from_client
from ray.util.client.server.server_stubs import current_server
def _use_response_cache(func):
    """
    Decorator for gRPC stubs. Before calling the real stubs, checks if there's
    an existing entry in the caches. If there is, then return the cached
    entry. Otherwise, call the real function and use the real cache
    """

    @functools.wraps(func)
    def wrapper(self, request, context):
        metadata = {k: v for k, v in context.invocation_metadata()}
        expected_ids = ('client_id', 'thread_id', 'req_id')
        if any((i not in metadata for i in expected_ids)):
            return func(self, request, context)
        client_id = metadata['client_id']
        thread_id = metadata['thread_id']
        req_id = int(metadata['req_id'])
        response_cache = self.response_caches[client_id]
        cached_entry = response_cache.check_cache(thread_id, req_id)
        if cached_entry is not None:
            if isinstance(cached_entry, Exception):
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details(str(cached_entry))
                raise cached_entry
            return cached_entry
        try:
            resp = func(self, request, context)
        except Exception as e:
            response_cache.update_cache(thread_id, req_id, e)
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(str(e))
            raise
        response_cache.update_cache(thread_id, req_id, resp)
        return resp
    return wrapper
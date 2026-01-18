import base64
import json
import logging
import os
import tempfile
import threading
import time
import uuid
import warnings
from collections import defaultdict
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import grpc
import ray._private.tls_utils
import ray.cloudpickle as cloudpickle
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray._private.ray_constants import DEFAULT_CLIENT_RECONNECT_GRACE_PERIOD
from ray._private.runtime_env.py_modules import upload_py_modules_if_needed
from ray._private.runtime_env.working_dir import upload_working_dir_if_needed
from ray.cloudpickle.compat import pickle
from ray.exceptions import GetTimeoutError
from ray.job_config import JobConfig
from ray.util.client.client_pickler import dumps_from_client, loads_from_server
from ray.util.client.common import (
from ray.util.client.dataclient import DataClient
from ray.util.client.logsclient import LogstreamClient
from ray.util.debug import log_once
def _add_ids_to_metadata(self, metadata: Any):
    """
        Adds a unique req_id and the current thread's identifier to the
        metadata. These values are useful for preventing mutating operations
        from being replayed on the server side in the event that the client
        must retry a requsest.
        Args:
            metadata - the gRPC metadata to append the IDs to
        """
    if not self._reconnect_enabled:
        return metadata
    thread_id = str(threading.get_ident())
    with self._req_id_lock:
        self._req_id += 1
        if self._req_id > INT32_MAX:
            self._req_id = 1
        req_id = str(self._req_id)
    return metadata + [('thread_id', thread_id), ('req_id', req_id)]
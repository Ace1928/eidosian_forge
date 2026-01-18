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
def _call_schedule_for_task(self, task: ray_client_pb2.ClientTask, num_returns: int) -> List[Future]:
    logger.debug(f'Scheduling task {task.name} {task.type} {task.payload_id}')
    task.client_id = self._client_id
    if num_returns is None:
        num_returns = 1
    num_return_refs = num_returns
    if num_return_refs == -1:
        num_return_refs = 1
    id_futures = [Future() for _ in range(num_return_refs)]

    def populate_ids(resp: Union[ray_client_pb2.DataResponse, Exception]) -> None:
        if isinstance(resp, Exception):
            if isinstance(resp, grpc.RpcError):
                resp = decode_exception(resp)
            for future in id_futures:
                future.set_exception(resp)
            return
        ticket = resp.task_ticket
        if not ticket.valid:
            try:
                ex = cloudpickle.loads(ticket.error)
            except (pickle.UnpicklingError, TypeError) as e_new:
                ex = e_new
            for future in id_futures:
                future.set_exception(ex)
            return
        if len(ticket.return_ids) != num_return_refs:
            exc = ValueError(f'Expected {num_return_refs} returns but received {len(ticket.return_ids)}')
            for future, raw_id in zip(id_futures, ticket.return_ids):
                future.set_exception(exc)
            return
        for future, raw_id in zip(id_futures, ticket.return_ids):
            future.set_result(raw_id)
    self.data_client.Schedule(task, populate_ids)
    self.total_outbound_message_size_bytes += task.ByteSize()
    if self.total_outbound_message_size_bytes > MESSAGE_SIZE_THRESHOLD and log_once('client_communication_overhead_warning'):
        warnings.warn(f"""More than 10MB of messages have been created to schedule tasks on the server. This can be slow on Ray Client due to communication overhead over the network. If you're running many fine-grained tasks, consider running them inside a single remote function. See the section on "Too fine-grained tasks" in the Ray Design Patterns document for more details: {DESIGN_PATTERN_FINE_GRAIN_TASKS_LINK}. If your functions frequently use large objects, consider storing the objects remotely with ray.put. An example of this is shown in the "Closure capture of large / unserializable object" section of the Ray Design Patterns document, available here: {DESIGN_PATTERN_LARGE_OBJECTS_LINK}""", UserWarning)
    return id_futures
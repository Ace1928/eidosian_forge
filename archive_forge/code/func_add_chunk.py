from collections import defaultdict
from ray.util.client.server.server_pickler import loads_from_client
import ray
import logging
import grpc
from queue import Queue
import sys
from typing import Any, Dict, Iterator, TYPE_CHECKING, Union
from threading import Event, Lock, Thread
import time
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray.util.client.common import (
from ray.util.client import CURRENT_PROTOCOL_VERSION
from ray.util.debug import log_once
from ray._private.client_mode_hook import disable_client_hook
def add_chunk(self, req: ray_client_pb2.DataRequest, chunk: Union[ray_client_pb2.PutRequest, ray_client_pb2.ClientTask]):
    if self.curr_req_id is not None and self.curr_req_id != req.req_id:
        raise RuntimeError(f'Expected to receive a chunk from request with id {self.curr_req_id}, but found {req.req_id} instead.')
    self.curr_req_id = req.req_id
    next_chunk = self.last_seen_chunk_id + 1
    if chunk.chunk_id < next_chunk:
        return
    if chunk.chunk_id > next_chunk:
        raise RuntimeError(f'A chunk {chunk.chunk_id} of request {req.req_id} was received out of order.')
    elif chunk.chunk_id == self.last_seen_chunk_id + 1:
        self.data.extend(chunk.data)
        self.last_seen_chunk_id = chunk.chunk_id
    return chunk.chunk_id + 1 == chunk.total_chunks
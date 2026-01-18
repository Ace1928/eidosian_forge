import math
import logging
import queue
import threading
import warnings
import grpc
from collections import OrderedDict
from typing import Any, Callable, Dict, TYPE_CHECKING, Optional, Union
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray.util.client.common import (
from ray.util.debug import log_once
def _process_response(self, response: Any) -> None:
    """
        Process responses from the data servicer.
        """
    if response.req_id == 0:
        logger.debug(f'Got unawaited response {response}')
        return
    if response.req_id in self.asyncio_waiting_data:
        can_remove = True
        try:
            callback = self.asyncio_waiting_data[response.req_id]
            if isinstance(callback, ChunkCollector):
                can_remove = callback(response)
            elif callback:
                callback(response)
            if can_remove:
                del self.asyncio_waiting_data[response.req_id]
        except Exception:
            logger.exception('Callback error:')
        with self.lock:
            if response.req_id in self.outstanding_requests and can_remove:
                del self.outstanding_requests[response.req_id]
                self._acknowledge(response.req_id)
    else:
        with self.lock:
            self.ready_data[response.req_id] = response
            self.cv.notify_all()